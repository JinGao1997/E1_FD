import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== 配置参数 ==========
root_dir = r"D:\PD_E1_UG_jg\EEG_R_Python_Pipeline_JG_Backup\E1_UG\export\n400_analysis"
full_label = {'dis': 'Disgust', 'dom': 'Dominance', 'neu': 'Neutral', 'aff': 'Affiliative', 'enj': 'Reward'}
emotion_colors = {'dis': "#755627", 'dom': "#F5900C", 'neu': "#C5C5C5EC", 'aff': "#39E04F", 'enj': "#FC0000"}
emotions = list(full_label.keys())
offer_types = ['fair', 'unfair']

# N400设置
erp = "N400"
erp_col = "N400"  # 数据表中N400成分的列名
erp_settings = {
    "roi": ["Fz", "Cz", "CPz", "Pz"],  # N400的ROI
    "win": (0.35, 0.50),   # N400时间窗350-500ms
    "ylabel": "N400 (µV)"
}

# ========== 第1步：标签提取 ==========
def add_labels_n400(input_path, output_path):
    """从ave.csv生成ave_with_labels.csv"""
    df = pd.read_csv(input_path)
    
    # 提取emotion（假设label格式为Face_enj_5_5）
    df['emotion'] = df['label'].str.split("_").str[1]
    
    # 提取offer_type
    df['offer_type'] = df['label'].apply(lambda x: (
        "fair" if x.split("_")[2:4] in [["5", "5"], ["6", "4"]] else
        "unfair" if x.split("_")[2:4] in [["8", "2"], ["9", "1"]] else None
    ))
    
    # 过滤掉7_3条件（如果存在）
    df = df[df['offer_type'].notna()]
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"标签提取完成，已保存至: {output_path}")
    return df

# ========== 第2步：数据读取和处理 ==========
data_file = os.path.join(root_dir, "ave.csv")
labeled_file = os.path.join(root_dir, "ave_with_labels.csv")
save_dir = os.path.join(root_dir, "figures_waveforms_N400")
os.makedirs(save_dir, exist_ok=True)

# 生成带标签的数据文件
if not os.path.exists(labeled_file):
    print("生成带标签的数据文件...")
    df = add_labels_n400(data_file, labeled_file)
else:
    print("读取带标签的数据文件...")
    df = pd.read_csv(labeled_file)

# 计算ROI均值
print(f"计算ROI均值，使用电极: {erp_settings['roi']}")
df['roi_mean'] = df[erp_settings['roi']].mean(axis=1)

# 按条件和时间计算均值
mean_df = df.groupby(['offer_type', 'emotion', 'time'])['roi_mean'].mean().reset_index().rename(columns={'roi_mean': 'mean'})

# 计算Y轴范围
sel = (mean_df['time']*1000 >= -200) & (mean_df['time']*1000 <= 1000)
ylim = (mean_df.loc[sel, 'mean'].min() - 0.3, mean_df.loc[sel, 'mean'].max() + 0.3)

# ========== 绘图函数 ==========
def plot_fair_vs_unfair_by_emotion(df, erp, ylab, win, ylim=None, save_dir=None):
    """每个emotion的Fair vs Unfair对比"""
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
    """单一条件下所有emotion对比"""
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
    """Fair vs Unfair总体平均"""
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

# ========== 批量绘图 ==========
print(f"\n开始绘制{erp}波形图...")
print(f"时间窗: {erp_settings['win'][0]*1000:.0f}-{erp_settings['win'][1]*1000:.0f}ms")
print(f"ROI: {', '.join(erp_settings['roi'])}")

# 绘制所有图形
plot_fair_vs_unfair_by_emotion(mean_df, erp, erp_settings['ylabel'], erp_settings['win'], ylim=ylim, save_dir=save_dir)
for offer_type in offer_types:
    plot_all_emotion_by_condition(mean_df, erp, erp_settings['ylabel'], erp_settings['win'], offer_type, ylim=ylim, save_dir=save_dir)
plot_fair_unfair_across_emotions(mean_df, erp, erp_settings['ylabel'], erp_settings['win'], ylim=ylim, save_dir=save_dir)

print(f"\n[{erp}] 波形图批量输出完成！")
print(f"所有图形已保存至: {save_dir}")