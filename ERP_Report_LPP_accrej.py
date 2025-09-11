import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== 配置参数 ==========
root_dir = r"D:\PD_E1_UG_jg\EEG_R_Python_Pipeline_JG_Backup\E1_UG\export\offer_lpp_accept_reject"
full_label = {'dis': 'Disgust', 'dom': 'Dominance', 'neu': 'Neutral', 'aff': 'Affiliative', 'enj': 'Reward'}
emotion_colors = {'dis': "#755627", 'dom': "#F5900C", 'neu': "#C5C5C5EC", 'aff': "#39E04F", 'enj': "#FC0000"}
emotions = list(full_label.keys())
offer_types = ['fair', 'unfair']
response_types = ['Accept', 'Reject']

# LPP设置（基于脚本2）
erp = "LPP"
erp_col = "LPP"  # 数据表中LPP成分的列名
erp_settings = {
    "roi": ["Pz", "Cz", "C1", "C2", "CP1", "CP2"],  # LPP的ROI
    "win": (0.4, 0.8),   # LPP时间窗400-800ms
    "ylabel": "LPP (µV)"
}

# 定义两种Y轴范围
ylim_narrow = (-5, 8)    # 用于特定图形
ylim_wide = (-12, 12)     # 用于其他图形

# ========== 第1步：标签提取 ==========
def add_labels_lpp(input_path, output_path):
    """从ave.csv生成ave_with_labels.csv，处理Accept/Reject"""
    df = pd.read_csv(input_path)
    
    # 提取emotion（假设label格式为Face_enj_5_5_Accept）
    df['emotion'] = df['label'].str.split("_").str[1]
    
    # 提取response type (Accept/Reject)
    df['response'] = df['label'].str.split("_").str[-1]
    
    # 提取offer_type
    def get_offer_type(label):
        parts = label.split("_")
        if len(parts) >= 5:
            offer_other, offer_you = parts[2], parts[3]
            if [offer_other, offer_you] in [["5", "5"], ["6", "4"]]:
                return "fair"
            elif [offer_other, offer_you] in [["8", "2"], ["9", "1"]]:
                return "unfair"
        return None
    
    df['offer_type'] = df['label'].apply(get_offer_type)
    
    # 过滤掉7_3条件
    df = df[df['offer_type'].notna()]
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"标签提取完成，已保存至: {output_path}")
    return df

# ========== 第2步：数据读取和处理 ==========
data_file = os.path.join(root_dir, "ave.csv")
labeled_file = os.path.join(root_dir, "ave_with_labels.csv")
save_dir = os.path.join(root_dir, "figures_waveforms_LPP")
os.makedirs(save_dir, exist_ok=True)

# 生成带标签的数据文件
if not os.path.exists(labeled_file):
    print("生成带标签的数据文件...")
    df = add_labels_lpp(data_file, labeled_file)
else:
    print("读取带标签的数据文件...")
    df = pd.read_csv(labeled_file)

# 计算ROI均值
print(f"计算ROI均值，使用电极: {erp_settings['roi']}")
df['roi_mean'] = df[erp_settings['roi']].mean(axis=1)

# 按条件和时间计算均值
mean_df = df.groupby(['offer_type', 'emotion', 'response', 'time'])['roi_mean'].mean().reset_index().rename(columns={'roi_mean': 'mean'})

print(f"Y轴范围设置：")
print(f"  - 窄范围 (-5, 8): FairnessXResponse, unfair_Reject, fair_Accept")
print(f"  - 宽范围 (-12, 12): 其他所有图形")

# ========== 原有绘图函数 ==========
def plot_accept_vs_reject_by_emotion(df, erp, ylab, win, ylim=None, save_dir=None):
    """每个emotion的Accept vs Reject对比"""
    for emo in emotions:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), facecolor='white')
        
        for idx, offer_type in enumerate(['fair', 'unfair']):
            ax = axes[idx]
            
            for response in response_types:
                d = df[(df['offer_type'] == offer_type) & (df['emotion'] == emo) & 
                       (df['response'] == response) & 
                       (df['time']*1000 >= -200) & (df['time']*1000 <= 1000)]
                if not d.empty:
                    linestyle = '-' if response == 'Accept' else '--'
                    ax.plot(d['time']*1000, d['mean'],
                           color=emotion_colors[emo], lw=1.5, linestyle=linestyle, label=response)
            
            ax.axvspan(win[0]*1000, win[1]*1000, color='grey', alpha=0.07, lw=0)
            ax.axvline(0, ls='-', color='#444444', lw=1)
            ax.axhline(0, ls=':', color="#9a9a9a", lw=1)
            ax.set_xlim([-200, 1000])
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.set_xlabel('Time (ms)', fontsize=8)
            ax.set_ylabel(ylab, fontsize=8)
            title = 'Fair' if offer_type == 'fair' else 'Unfair'
            ax.set_title(f'{erp}: {full_label[emo]} ({title})', fontsize=10, weight='bold')
            ax.tick_params(axis='both', labelsize=8)
            ax.legend(fontsize=8, frameon=True, fancybox=True)
        
        plt.tight_layout()
        if save_dir:
            fname = os.path.join(save_dir, f"{erp}_{emo}_AcceptVsReject_Waveform.tif")
            plt.savefig(fname, dpi=400, bbox_inches='tight', format='tiff')
            print(f"  已保存: {fname}")
        plt.close(fig)

def plot_fair_unfair_accept_reject(df, erp, ylab, win, ylim=None, save_dir=None):
    """Fair vs Unfair × Accept vs Reject交互图"""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    
    colors = {'fair': '#4CAF50', 'unfair': '#F44336'}
    styles = {'Accept': '-', 'Reject': '--'}
    
    for offer_type in offer_types:
        for response in response_types:
            # 对所有emotion求平均
            d = df[(df['offer_type'] == offer_type) & (df['response'] == response)]
            mean_data = d.groupby('time')['mean'].mean().reset_index()
            
            label = f"{offer_type.capitalize()} - {response}"
            ax.plot(mean_data['time']*1000, mean_data['mean'],
                   color=colors[offer_type], linestyle=styles[response],
                   lw=1.8, label=label)
    
    ax.axvspan(win[0]*1000, win[1]*1000, color='grey', alpha=0.07, lw=0)
    ax.axvline(0, ls='-', color='#444444', lw=1)
    ax.axhline(0, ls=':', color="#9a9a9a", lw=1)
    ax.set_xlim([-200, 1000])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_title(f'{erp}: Fairness × Response Interaction', fontsize=12, weight='bold')
    ax.tick_params(axis='both', labelsize=9)
    ax.legend(fontsize=9, frameon=True, fancybox=True, loc='upper right')
    
    plt.tight_layout()
    if save_dir:
        fname = os.path.join(save_dir, f"{erp}_FairnessXResponse_Waveform.tif")
        plt.savefig(fname, dpi=400, bbox_inches='tight', format='tiff')
        print(f"  已保存: {fname}")
    plt.close(fig)

def plot_emotion_grid(df, erp, ylab, win, ylim=None, save_dir=None):
    """5×2网格图：每个emotion的Fair/Unfair × Accept/Reject"""
    fig, axes = plt.subplots(5, 2, figsize=(10, 12), facecolor='white')
    
    for row, emo in enumerate(emotions):
        for col, offer_type in enumerate(offer_types):
            ax = axes[row, col]
            
            for response in response_types:
                d = df[(df['offer_type'] == offer_type) & (df['emotion'] == emo) & 
                       (df['response'] == response) & 
                       (df['time']*1000 >= -200) & (df['time']*1000 <= 1000)]
                if not d.empty:
                    linestyle = '-' if response == 'Accept' else '--'
                    ax.plot(d['time']*1000, d['mean'],
                           color=emotion_colors[emo], lw=1.2, linestyle=linestyle, label=response)
            
            ax.axvspan(win[0]*1000, win[1]*1000, color='grey', alpha=0.07, lw=0)
            ax.axvline(0, ls='-', color='#444444', lw=0.8)
            ax.axhline(0, ls=':', color="#9a9a9a", lw=0.8)
            ax.set_xlim([-200, 1000])
            if ylim is not None:
                ax.set_ylim(ylim)
            
            if row == 4:
                ax.set_xlabel('Time (ms)', fontsize=8)
            if col == 0:
                ax.set_ylabel(f'{full_label[emo]}\n{ylab}', fontsize=8)
            if row == 0:
                title = 'Fair' if offer_type == 'fair' else 'Unfair'
                ax.set_title(title, fontsize=10, weight='bold')
            
            ax.tick_params(axis='both', labelsize=7)
            if row == 0 and col == 1:
                ax.legend(fontsize=7, frameon=True, loc='upper right')
    
    plt.suptitle(f'{erp} Waveforms: All Conditions', fontsize=14, weight='bold', y=0.995)
    plt.tight_layout()
    if save_dir:
        fname = os.path.join(save_dir, f"{erp}_AllConditions_Grid.tif")
        plt.savefig(fname, dpi=400, bbox_inches='tight', format='tiff')
        print(f"  已保存: {fname}")
    plt.close(fig)

# ========== 新增绘图函数 ==========
def plot_emotions_by_fairness_accept_only(df, erp, ylab, win, offer_type, ylim=None, save_dir=None):
    """固定fairness条件，仅显示Accept的所有emotions对比"""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    
    for emo in emotions:
        d = df[(df['offer_type'] == offer_type) & (df['emotion'] == emo) & 
               (df['response'] == 'Accept') & 
               (df['time']*1000 >= -200) & (df['time']*1000 <= 1000)]
        if not d.empty:
            ax.plot(d['time']*1000, d['mean'],
                   color=emotion_colors[emo], lw=1.8, 
                   linestyle='-', label=full_label[emo])
    
    ax.axvspan(win[0]*1000, win[1]*1000, color='grey', alpha=0.07, lw=0)
    ax.axvline(0, ls='-', color='#444444', lw=1)
    ax.axhline(0, ls=':', color="#9a9a9a", lw=1)
    ax.set_xlim([-200, 1000])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)
    title_str = offer_type.capitalize()
    ax.set_title(f'{erp}: Accept Responses Only ({title_str} Offers)', 
                fontsize=12, weight='bold')
    ax.tick_params(axis='both', labelsize=9)
    ax.legend(fontsize=9, frameon=True, fancybox=True, loc='upper right')
    
    plt.tight_layout()
    if save_dir:
        fname = os.path.join(save_dir, f"{erp}_{offer_type}_Accept_AllEmotions.tif")
        plt.savefig(fname, dpi=400, bbox_inches='tight', format='tiff')
        print(f"  已保存: {fname}")
    plt.close(fig)

def plot_emotions_by_fairness_reject_only(df, erp, ylab, win, offer_type, ylim=None, save_dir=None):
    """固定fairness条件，仅显示Reject的所有emotions对比"""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    
    for emo in emotions:
        d = df[(df['offer_type'] == offer_type) & (df['emotion'] == emo) & 
               (df['response'] == 'Reject') & 
               (df['time']*1000 >= -200) & (df['time']*1000 <= 1000)]
        if not d.empty:
            ax.plot(d['time']*1000, d['mean'],
                   color=emotion_colors[emo], lw=1.8, 
                   linestyle='--', label=full_label[emo])
    
    ax.axvspan(win[0]*1000, win[1]*1000, color='grey', alpha=0.07, lw=0)
    ax.axvline(0, ls='-', color='#444444', lw=1)
    ax.axhline(0, ls=':', color="#9a9a9a", lw=1)
    ax.set_xlim([-200, 1000])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)
    title_str = offer_type.capitalize()
    ax.set_title(f'{erp}: Reject Responses Only ({title_str} Offers)', 
                fontsize=12, weight='bold')
    ax.tick_params(axis='both', labelsize=9)
    ax.legend(fontsize=9, frameon=True, fancybox=True, loc='upper right')
    
    plt.tight_layout()
    if save_dir:
        fname = os.path.join(save_dir, f"{erp}_{offer_type}_Reject_AllEmotions.tif")
        plt.savefig(fname, dpi=400, bbox_inches='tight', format='tiff')
        print(f"  已保存: {fname}")
    plt.close(fig)

# ========== 批量绘图 ==========
print(f"\n开始绘制{erp}波形图...")
print(f"时间窗: {erp_settings['win'][0]*1000:.0f}-{erp_settings['win'][1]*1000:.0f}ms")
print(f"ROI: {', '.join(erp_settings['roi'])}")

# 绘制原有图形
print("\n绘制原有图形...")
# 使用宽范围的图形
plot_accept_vs_reject_by_emotion(mean_df, erp, erp_settings['ylabel'], 
                                 erp_settings['win'], ylim=ylim_wide, save_dir=save_dir)
plot_emotion_grid(mean_df, erp, erp_settings['ylabel'], 
                 erp_settings['win'], ylim=ylim_wide, save_dir=save_dir)

# 使用窄范围的图形
plot_fair_unfair_accept_reject(mean_df, erp, erp_settings['ylabel'], 
                               erp_settings['win'], ylim=ylim_narrow, save_dir=save_dir)

# 绘制新增图形
print("\n绘制新增图形...")
# Unfair条件
plot_emotions_by_fairness_accept_only(mean_df, erp, erp_settings['ylabel'], 
                                      erp_settings['win'], 'unfair', ylim=ylim_wide, save_dir=save_dir)
plot_emotions_by_fairness_reject_only(mean_df, erp, erp_settings['ylabel'], 
                                      erp_settings['win'], 'unfair', ylim=ylim_narrow, save_dir=save_dir)

# Fair条件
plot_emotions_by_fairness_accept_only(mean_df, erp, erp_settings['ylabel'], 
                                      erp_settings['win'], 'fair', ylim=ylim_narrow, save_dir=save_dir)
plot_emotions_by_fairness_reject_only(mean_df, erp, erp_settings['ylabel'], 
                                      erp_settings['win'], 'fair', ylim=ylim_wide, save_dir=save_dir)

print(f"\n[{erp}] 波形图批量输出完成！共生成7种类型的图形")
print(f"所有图形已保存至: {save_dir}")