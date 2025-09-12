import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# =============== 配置参数 / Configuration =================
# =========================================================
root_dir = r"D:\PD_E1_UG_jg\EEG_R_Python_Pipeline_JG_Backup\E1_UG\export\n400_analysis"

full_label = {'dis': 'Disgust', 'dom': 'Dominance', 'neu': 'Neutral', 'aff': 'Affiliative', 'enj': 'Reward'}
emotion_colors = {'dis': "#755627", 'dom': "#F5900C", 'neu': "#C5C5C5EC", 'aff': "#39E04F", 'enj': "#FC0000"}
emotions = list(full_label.keys())
offer_types = ['fair', 'unfair']

# —— 参考情绪（差异波 emo - REF_EMOTION），此处已切换为 'enj'（Reward）
REF_EMOTION = 'enj'  # 可改为 'neu' 或其它情绪，以便快速切换参考

# N400设置
erp = "N400"
erp_col = "N400"  # 如果后续需要按列名筛选或计算，可保留
erp_settings = {
    "roi": ["Fz", "Cz", "CPz", "Pz"],  # N400的ROI
    "win": (0.35, 0.450),              # N400时间窗350-450ms
    "ylabel": "N400 (µV)"
}

# =========================================================
# ===================== 第1步：标签提取 =====================
# =========================================================
def add_labels_n400(input_path, output_path):
    """从 ave.csv 生成 ave_with_labels.csv，并抽取 emotion / offer_type 标签"""
    df = pd.read_csv(input_path, encoding='utf-8')
    
    # 提取 emotion（假设 label 格式为 Face_enj_5_5）
    df['emotion'] = df['label'].str.split("_").str[1]
    
    # 提取 offer_type
    df['offer_type'] = df['label'].apply(lambda x: (
        "fair" if x.split("_")[2:4] in [["5", "5"], ["6", "4"]] else
        "unfair" if x.split("_")[2:4] in [["8", "2"], ["9", "1"]] else None
    ))
    
    # 过滤掉 7_3 条件（如果存在）
    df = df[df['offer_type'].notna()]
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"标签提取完成，已保存至: {output_path}")
    return df

# =========================================================
# ================= 第2步：数据读取和处理 ===================
# =========================================================
data_file = os.path.join(root_dir, "ave.csv")
labeled_file = os.path.join(root_dir, "ave_with_labels.csv")
save_dir = os.path.join(root_dir, "figures_waveforms_N400")
os.makedirs(save_dir, exist_ok=True)

if not os.path.exists(labeled_file):
    print("生成带标签的数据文件...")
    df = add_labels_n400(data_file, labeled_file)
else:
    print("读取带标签的数据文件...")
    df = pd.read_csv(labeled_file, encoding='utf-8')

# 计算 ROI 均值
print(f"计算ROI均值，使用电极: {erp_settings['roi']}")
df['roi_mean'] = df[erp_settings['roi']].mean(axis=1)

# 按条件与时间聚合
mean_df = df.groupby(['offer_type', 'emotion', 'time'])['roi_mean'].mean().reset_index().rename(columns={'roi_mean': 'mean'})

# 原始波形 Y 轴范围（-200~1000ms）
sel = (mean_df['time']*1000 >= -200) & (mean_df['time']*1000 <= 1000)
ylim = (mean_df.loc[sel, 'mean'].min() - 0.3, mean_df.loc[sel, 'mean'].max() + 0.3)

# =========================================================
# ========== 计算差异波（统一Y轴范围，参考=REF_EMOTION） =========
# =========================================================
def calculate_diff_ylim(mean_df, margin=0.15):
    """计算所有差异波的统一Y轴范围（将参考情绪设为 REF_EMOTION）"""
    all_diffs = []
    
    # 1) Unfair - Fair（与参考无关，保持不变）
    for emo in emotions:
        fair_data = mean_df[(mean_df['offer_type'] == 'fair') & (mean_df['emotion'] == emo)]
        unfair_data = mean_df[(mean_df['offer_type'] == 'unfair') & (mean_df['emotion'] == emo)]
        if not fair_data.empty and not unfair_data.empty:
            merged = pd.merge(
                unfair_data[['time', 'mean']],
                fair_data[['time', 'mean']],
                on='time', suffixes=('_unfair', '_fair')
            )
            merged['diff'] = merged['mean_unfair'] - merged['mean_fair']
            merged = merged[(merged['time']*1000 >= -200) & (merged['time']*1000 <= 1000)]
            all_diffs.extend(merged['diff'].values)
    
    # 2) emotions - REF_EMOTION（分条件）
    emotions_vs_ref = [e for e in emotions if e != REF_EMOTION]
    for offer_type in offer_types:
        ref_data = mean_df[(mean_df['offer_type'] == offer_type) & (mean_df['emotion'] == REF_EMOTION)]
        if not ref_data.empty:
            for emo in emotions_vs_ref:
                emo_data = mean_df[(mean_df['offer_type'] == offer_type) & (mean_df['emotion'] == emo)]
                if not emo_data.empty:
                    merged = pd.merge(
                        emo_data[['time', 'mean']],
                        ref_data[['time', 'mean']],
                        on='time', suffixes=('_emo', '_ref')
                    )
                    merged['diff'] = merged['mean_emo'] - merged['mean_ref']  # emo - REF
                    merged = merged[(merged['time']*1000 >= -200) & (merged['time']*1000 <= 1000)]
                    all_diffs.extend(merged['diff'].values)
    
    # 3) emotions - REF_EMOTION（跨 offer 平均）
    for emo in emotions_vs_ref:
        emo_avg = mean_df[mean_df['emotion'] == emo].groupby('time')['mean'].mean().reset_index()
        ref_avg = mean_df[mean_df['emotion'] == REF_EMOTION].groupby('time')['mean'].mean().reset_index()
        merged = pd.merge(emo_avg, ref_avg, on='time', suffixes=('_emo', '_ref'))
        merged = merged[(merged['time']*1000 >= -200) & (merged['time']*1000 <= 1000)]
        merged['diff'] = merged['mean_emo'] - merged['mean_ref']  # emo - REF
        all_diffs.extend(merged['diff'].values)
    
    if all_diffs:
        min_val = float(np.min(all_diffs))
        max_val = float(np.max(all_diffs))
        y_range = max_val - min_val
        ylim_diff = (min_val - margin * y_range, max_val + margin * y_range)
        print(f"差异波Y轴范围计算完成（参考={full_label[REF_EMOTION]}）: [{ylim_diff[0]:.2f}, {ylim_diff[1]:.2f}] µV")
        return ylim_diff
    else:
        print("警告：无法计算差异波范围，使用默认值")
        return (-2.0, 2.0)

ylim_diff = calculate_diff_ylim(mean_df, margin=0.15)

# =========================================================
# ======================== 绘图函数 ========================
# =========================================================
def plot_fair_vs_unfair_by_emotion(df, erp, ylab, win, ylim=None, save_dir=None):
    """每个 emotion 的 Fair vs Unfair 对比"""
    for emo in emotions:
        fig, ax = plt.subplots(figsize=(7, 4.2), facecolor='white')
        for offer_type in offer_types:
            d = df[(df['offer_type'] == offer_type) & (df['emotion'] == emo) &
                   (df['time']*1000 >= -200) & (df['time']*1000 <= 1000)]
            if not d.empty:
                linestyle = '-.' if offer_type == 'fair' else '-'
                lw = 1.5
                label = 'Fair' if offer_type == 'fair' else 'Unfair'
                ax.plot(d['time']*1000, d['mean'],
                        color=emotion_colors[emo], lw=lw, linestyle=linestyle, label=label)
        
        ax.axvspan(win[0]*1000, win[1]*1000, color='grey', alpha=0.07, lw=0)
        ax.axvline(0, ls='-', color='#444444', lw=1)
        ax.axhline(0, ls=':', color="#9a9a9a", lw=1)
        ax.set_xlim([-200, 1000])
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlabel('Time (ms)', fontsize=8)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_title(f'{erp} ROI: {full_label[emo]} (Fair vs Unfair)', fontsize=10, weight='bold', pad=4)
        ax.set_xticks(np.arange(-200, 1100, 100))
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        leg = ax.legend(
            title="Condition", fontsize=8, title_fontsize=9,
            frameon=True, fancybox=True, ncol=1, framealpha=0.85,
            loc='center left', bbox_to_anchor=(1.00, 0.52), borderaxespad=0.5, handlelength=1.5
        )
        leg.get_frame().set_edgecolor('lightgrey')
        plt.tight_layout(rect=[0, 0, 0.98, 1])
        if save_dir:
            fname = os.path.join(save_dir, f"{erp}_{emo}_FairVsUnfair_Waveform.tif")
            plt.savefig(fname, dpi=400, bbox_inches='tight', format='tiff')
            print(f"  已保存: {fname}")
        plt.close(fig)

def plot_all_emotion_by_condition(df, erp, ylab, win, offer_type, ylim=None, save_dir=None):
    """单一条件下所有 emotion 的对比（Fair 或 Unfair）"""
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor='white')
    linestyle = '-.' if offer_type == 'fair' else '-'
    lw = 1.5
    for emo in emotions:
        d = df[(df['offer_type'] == offer_type) & (df['emotion'] == emo) &
               (df['time']*1000 >= -200) & (df['time']*1000 <= 1000)]
        if not d.empty:
            ax.plot(d['time']*1000, d['mean'],
                    color=emotion_colors[emo], lw=lw, linestyle=linestyle,
                    label=full_label[emo])
    
    ax.axvspan(win[0]*1000, win[1]*1000, color='grey', alpha=0.07, lw=0)
    ax.axvline(0, ls='-', color='#444444', lw=1)
    ax.axhline(0, ls=':', color="#9a9a9a", lw=1)
    ax.set_xlim([-200, 1000])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Time (ms)', fontsize=8)
    ax.set_ylabel(ylab, fontsize=8)
    title_cn = 'Fair' if offer_type == 'fair' else 'Unfair'
    ax.set_title(f'{erp} ROI: All Emotions ({title_cn})', fontsize=10, weight='bold', pad=4)
    ax.set_xticks(np.arange(-200, 1100, 100))
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    leg = ax.legend(
        title="Emotion", fontsize=8, title_fontsize=9,
        frameon=True, fancybox=True, ncol=1, framealpha=0.85,
        loc='center left', bbox_to_anchor=(1.00, 0.52), borderaxespad=0.5, handlelength=1.5
    )
    leg.get_frame().set_edgecolor('lightgrey')
    plt.tight_layout(rect=[0, 0, 0.98, 1])
    if save_dir:
        fname = os.path.join(save_dir, f"{erp}_{offer_type}_AllEmo_Waveform.tif")
        plt.savefig(fname, dpi=400, bbox_inches='tight', format='tiff')
        print(f"  已保存: {fname}")
    plt.close(fig)

def plot_fair_unfair_across_emotions(df, erp, ylab, win, ylim=None, save_dir=None):
    """Fair vs Unfair（跨所有情绪平均）"""
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor='white')
    for offer_type in offer_types:
        d = df[df['offer_type'] == offer_type]
        mean_over_emo = d.groupby('time')['mean'].mean().reset_index()
        style = '-.' if offer_type == 'fair' else '-'
        lw = 1.5
        label = 'Fair' if offer_type == 'fair' else 'Unfair'
        ax.plot(mean_over_emo['time']*1000, mean_over_emo['mean'],
                color='black', lw=lw, linestyle=style, label=label)
    
    ax.axvspan(win[0]*1000, win[1]*1000, color='grey', alpha=0.07, lw=0)
    ax.axvline(0, ls='-', color='#444444', lw=1)
    ax.axhline(0, ls=':', color="#9a9a9a", lw=1)
    ax.set_xlim([-200, 1000])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Time (ms)', fontsize=8)
    ax.set_ylabel(ylab, fontsize=8)
    ax.set_title(f'{erp} ROI: Fair vs Unfair (Averaged Across Emotions)', fontsize=10, weight='bold', pad=4)
    ax.set_xticks(np.arange(-200, 1100, 100))
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    leg = ax.legend(
        title="Condition", fontsize=8, title_fontsize=9,
        frameon=True, fancybox=True, ncol=1, framealpha=0.85,
        loc='center left', bbox_to_anchor=(1.00, 0.52), borderaxespad=0.5, handlelength=1.5
    )
    leg.get_frame().set_edgecolor('lightgrey')
    plt.tight_layout(rect=[0, 0, 0.98, 1])
    if save_dir:
        fname = os.path.join(save_dir, f"{erp}_FairVsUnfair_AllMean_Waveform.tif")
        plt.savefig(fname, dpi=400, bbox_inches='tight', format='tiff')
        print(f"  已保存: {fname}")
    plt.close(fig)

# ------------------------ 差异波 -------------------------
def plot_unfair_minus_fair_by_emotion(df, erp, ylab, win, ylim_diff=None, save_dir=None):
    """每个 emotion 的 (Unfair - Fair) 差异波"""
    for emo in emotions:
        fig, ax = plt.subplots(figsize=(7, 4.2), facecolor='white')
        fair_data = df[(df['offer_type'] == 'fair') & (df['emotion'] == emo) &
                       (df['time']*1000 >= -200) & (df['time']*1000 <= 1000)]
        unfair_data = df[(df['offer_type'] == 'unfair') & (df['emotion'] == emo) &
                         (df['time']*1000 >= -200) & (df['time']*1000 <= 1000)]
        if not fair_data.empty and not unfair_data.empty:
            merged = pd.merge(
                unfair_data[['time', 'mean']],
                fair_data[['time', 'mean']],
                on='time', suffixes=('_unfair', '_fair')
            )
            merged['diff'] = merged['mean_unfair'] - merged['mean_fair']
            ax.plot(merged['time']*1000, merged['diff'],
                    color=emotion_colors[emo], lw=2, linestyle='-',
                    label=f'{full_label[emo]} (Unfair - Fair)')
        
        ax.axvspan(win[0]*1000, win[1]*1000, color='grey', alpha=0.07, lw=0)
        ax.axvline(0, ls='-', color='#444444', lw=1)
        ax.axhline(0, ls=':', color="#9a9a9a", lw=1)
        ax.set_xlim([-200, 1000])
        if ylim_diff is not None:
            ax.set_ylim(ylim_diff)
        ax.set_xlabel('Time (ms)', fontsize=8)
        ax.set_ylabel(f'{ylab} (Difference)', fontsize=8)
        ax.set_title(f'{erp} ROI: {full_label[emo]} (Unfair - Fair)', fontsize=10, weight='bold', pad=4)
        ax.set_xticks(np.arange(-200, 1100, 100))
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        leg = ax.legend(
            fontsize=8, frameon=True, fancybox=True, ncol=1, framealpha=0.85,
            loc='center left', bbox_to_anchor=(1.00, 0.52), borderaxespad=0.5, handlelength=1.5
        )
        leg.get_frame().set_edgecolor('lightgrey')
        plt.tight_layout(rect=[0, 0, 0.98, 1])
        if save_dir:
            fname = os.path.join(save_dir, f"{erp}_{emo}_UnfairMinusFair_DiffWave.tif")
            plt.savefig(fname, dpi=400, bbox_inches='tight', format='tiff')
            print(f"  已保存: {fname}")
        plt.close(fig)

def plot_emotion_minus_ref_by_condition(df, erp, ylab, win, ylim_diff=None, save_dir=None):
    """绘制各情绪减去参考情绪 REF_EMOTION 的差异波（分 Fair/Unfair 条件）"""
    emotions_vs_ref = [e for e in emotions if e != REF_EMOTION]
    for offer_type in offer_types:
        fig, ax = plt.subplots(figsize=(7, 4.2), facecolor='white')
        ref_data = df[(df['offer_type'] == offer_type) & (df['emotion'] == REF_EMOTION) &
                      (df['time']*1000 >= -200) & (df['time']*1000 <= 1000)]
        if not ref_data.empty:
            for emo in emotions_vs_ref:
                emo_data = df[(df['offer_type'] == offer_type) & (df['emotion'] == emo) &
                              (df['time']*1000 >= -200) & (df['time']*1000 <= 1000)]
                if not emo_data.empty:
                    merged = pd.merge(
                        emo_data[['time', 'mean']],
                        ref_data[['time', 'mean']],
                        on='time', suffixes=('_emo', '_ref')
                    )
                    merged['diff'] = merged['mean_emo'] - merged['mean_ref']  # emo - REF
                    ax.plot(
                        merged['time']*1000, merged['diff'],
                        color=emotion_colors[emo], lw=1.8, linestyle='-',
                        label=f"{full_label[emo]} - {full_label[REF_EMOTION]}"
                    )
        
        ax.axvspan(win[0]*1000, win[1]*1000, color='grey', alpha=0.07, lw=0)
        ax.axvline(0, ls='-', color='#444444', lw=1)
        ax.axhline(0, ls=':', color="#9a9a9a", lw=1)
        ax.set_xlim([-200, 1000])
        if ylim_diff is not None:
            ax.set_ylim(ylim_diff)
        ax.set_xlabel('Time (ms)', fontsize=8)
        ax.set_ylabel(f'{ylab} (Difference)', fontsize=8)
        title_cn = 'Fair' if offer_type == 'fair' else 'Unfair'
        ax.set_title(f'{erp} ROI: Emotions vs {full_label[REF_EMOTION]} ({title_cn})', fontsize=10, weight='bold', pad=4)
        ax.set_xticks(np.arange(-200, 1100, 100))
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        leg = ax.legend(
            title="Difference", fontsize=8, title_fontsize=9,
            frameon=True, fancybox=True, ncol=1, framealpha=0.85,
            loc='center left', bbox_to_anchor=(1.00, 0.52), borderaxespad=0.5, handlelength=1.5
        )
        leg.get_frame().set_edgecolor('lightgrey')
        plt.tight_layout(rect=[0, 0, 0.98, 1])
        if save_dir:
            fname = os.path.join(save_dir, f"{erp}_{offer_type}_EmotionsMinus{full_label[REF_EMOTION]}_DiffWave.tif")
            plt.savefig(fname, dpi=400, bbox_inches='tight', format='tiff')
            print(f"  已保存: {fname}")
        plt.close(fig)

def plot_emotion_minus_ref_averaged(df, erp, ylab, win, ylim_diff=None, save_dir=None):
    """绘制各情绪减去参考情绪 REF_EMOTION 的差异波（跨 Fair/Unfair 平均）"""
    emotions_vs_ref = [e for e in emotions if e != REF_EMOTION]
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor='white')
    for emo in emotions_vs_ref:
        emo_avg = df[df['emotion'] == emo].groupby('time')['mean'].mean().reset_index()
        ref_avg = df[df['emotion'] == REF_EMOTION].groupby('time')['mean'].mean().reset_index()
        merged = pd.merge(emo_avg, ref_avg, on='time', suffixes=('_emo', '_ref'))
        merged = merged[(merged['time']*1000 >= -200) & (merged['time']*1000 <= 1000)]
        merged['diff'] = merged['mean_emo'] - merged['mean_ref']  # emo - REF
        ax.plot(
            merged['time']*1000, merged['diff'],
            color=emotion_colors[emo], lw=2, linestyle='-',
            label=f"{full_label[emo]} - {full_label[REF_EMOTION]}"
        )
    
    ax.axvspan(win[0]*1000, win[1]*1000, color='grey', alpha=0.07, lw=0)
    ax.axvline(0, ls='-', color='#444444', lw=1)
    ax.axhline(0, ls=':', color="#9a9a9a", lw=1)
    ax.set_xlim([-200, 1000])
    if ylim_diff is not None:
        ax.set_ylim(ylim_diff)
    ax.set_xlabel('Time (ms)', fontsize=8)
    ax.set_ylabel(f'{ylab} (Difference)', fontsize=8)
    ax.set_title(f'{erp} ROI: Emotions vs {full_label[REF_EMOTION]} (Averaged Across Offers)', fontsize=10, weight='bold', pad=4)
    ax.set_xticks(np.arange(-200, 1100, 100))
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    leg = ax.legend(
        title="Difference", fontsize=8, title_fontsize=9,
        frameon=True, fancybox=True, ncol=1, framealpha=0.85,
        loc='center left', bbox_to_anchor=(1.00, 0.52), borderaxespad=0.5, handlelength=1.5
    )
    leg.get_frame().set_edgecolor('lightgrey')
    plt.tight_layout(rect=[0, 0, 0.98, 1])
    if save_dir:
        fname = os.path.join(save_dir, f"{erp}_EmotionsMinus{full_label[REF_EMOTION]}_Averaged_DiffWave.tif")
        plt.savefig(fname, dpi=400, bbox_inches='tight', format='tiff')
        print(f"  已保存: {fname}")
    plt.close(fig)

# =========================================================
# ======================== 批量绘图 ========================
# =========================================================
print(f"\n开始绘制{erp}波形图...")
print(f"时间窗: {erp_settings['win'][0]*1000:.0f}-{erp_settings['win'][1]*1000:.0f}ms")
print(f"ROI: {', '.join(erp_settings['roi'])}")
print(f"原始波形Y轴范围: [{ylim[0]:.2f}, {ylim[1]:.2f}] µV")
print(f"差异波Y轴范围（参考={full_label[REF_EMOTION]}）: [{ylim_diff[0]:.2f}, {ylim_diff[1]:.2f}] µV")

# —— 原始波形图
print("\n绘制原始波形图...")
plot_fair_vs_unfair_by_emotion(mean_df, erp, erp_settings['ylabel'], erp_settings['win'], ylim=ylim, save_dir=save_dir)
for offer_type in offer_types:
    plot_all_emotion_by_condition(mean_df, erp, erp_settings['ylabel'], erp_settings['win'], offer_type, ylim=ylim, save_dir=save_dir)
plot_fair_unfair_across_emotions(mean_df, erp, erp_settings['ylabel'], erp_settings['win'], ylim=ylim, save_dir=save_dir)

# —— 差异波形图
print("\n绘制差异波形图...")
# 1) Unfair - Fair（每个情绪）
plot_unfair_minus_fair_by_emotion(mean_df, erp, erp_settings['ylabel'], erp_settings['win'],
                                  ylim_diff=ylim_diff, save_dir=save_dir)

# 2) 各情绪 - 参考情绪（分条件）
plot_emotion_minus_ref_by_condition(mean_df, erp, erp_settings['ylabel'], erp_settings['win'],
                                    ylim_diff=ylim_diff, save_dir=save_dir)

# 3) 各情绪 - 参考情绪（跨条件平均）
plot_emotion_minus_ref_averaged(mean_df, erp, erp_settings['ylabel'], erp_settings['win'],
                                ylim_diff=ylim_diff, save_dir=save_dir)

print(f"\n[{erp}] 所有波形图（含差异波）批量输出完成！")
print(f"所有图形已保存至: {save_dir}")
print(f"- 原始波形图: 8张")
print(f"- Unfair-Fair差异波: {len(emotions)}张")  
print(f"- Emotions-{full_label[REF_EMOTION]}差异波: {len(offer_types) + 1}张")
print(f"总计: {8 + len(emotions) + len(offer_types) + 1}张图形")
