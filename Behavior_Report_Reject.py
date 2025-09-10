from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 防止Win下Tk后端崩溃，必须放在最顶端！
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 全局配置 ==========

analysis_types = ["ratio", "type"]
base_dir = Path().cwd()
results_root = base_dir / "analysis_output" / "results_refuse"

full_label = {
    'dis': 'Disgust',
    'dom': 'Dominance',
    'neu': 'Neutral',
    'aff': 'Affiliative',
    'enj': 'Reward'
}
emotion_order = ["dis", "dom", "neu", "aff", "enj"]
emotion_colors = {
    'dis': "#8c510a",
    'dom': "#e08214",
    'neu': "#bababa",
    'aff': "#39E04F",
    'enj': "#d73027"
}
def rgba(hex_color, alpha=0.36):
    import matplotlib.colors as mcolors
    rgb = mcolors.hex2color(hex_color)
    return tuple(list(rgb) + [alpha])
custom_palette = [rgba(emotion_colors[e]) for e in emotion_order]
edge_palette = [emotion_colors[e] for e in emotion_order]

ratio_levels = ["5:5", "4:6", "3:7", "2:8", "1:9"]
type_levels = ["fair", "unfair"]
CI = 1.96

for analysis_type in analysis_types:
    print(f"\n=== 绘制【{analysis_type}】分析的拒绝率可视化 ===")
    results_dir = results_root / analysis_type
    results_dir.mkdir(parents=True, exist_ok=True)

    # ====== 读取统计分析结果 ======
    glmm_main = pd.read_csv(results_dir / "glmm_main_effects.csv")
    pred_glmm = pd.read_csv(results_dir / "glmm_interaction_preds.csv")
    lmm_main  = pd.read_csv(results_dir / "lmm_main_effects.csv")
    pred_lmm  = pd.read_csv(results_dir / "lmm_interaction_preds.csv")

    # 选择横轴类型
    if analysis_type == "ratio":
        x_col = "offer_ratio"
        x_levels = [r for r in ratio_levels if r in pred_glmm[x_col].unique()]
    else:
        x_col = "offer_type"
        x_levels = [r for r in type_levels if r in pred_glmm[x_col].unique()]

    def normalize_ratio(s):
        if ':' in str(s):
            a, b = s.split(':')
            return f"{int(a)}:{int(b)}"
        else:
            return s
    for df in (pred_glmm, pred_lmm):
        df['x_norm'] = df[x_col].apply(normalize_ratio)

    # ========== 主效应barplot ==========
    emo_glmm = glmm_main[glmm_main['effect']=="Emotion"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7,4))
    for i, row in emo_glmm.iterrows():
        code = row['contrast'].split()[0]
        or_  = row['odds.ratio']
        err  = CI * row['SE']
        ax.bar(i, or_, yerr=err,
               color=emotion_colors.get(code, "#cccccc"),
               edgecolor='black', capsize=4)
    ax.axhline(1, linestyle='--', color='gray')
    ax.set_xticks(np.arange(len(emo_glmm)))
    ax.set_xticklabels([full_label.get(c.split()[0], c) for c in emo_glmm['contrast']], rotation=15)
    ax.set_ylabel('Odds Ratio (Refusal)')
    ax.set_title(f'Rejection Rate: Main Effects (Emotion) [{analysis_type}]')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(results_dir / "rejection_glmm_main_emotion.tiff", dpi=220)
    plt.close(fig)

    # ========== 交互barplot ==========
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.15
    for idx, code in enumerate(emotion_order):
        df = (pred_glmm[pred_glmm['emotion'] == code]
              .set_index('x_norm').reindex(x_levels).reset_index())
        if df.empty: continue
        x_pos = np.arange(len(x_levels)) + (idx - 2) * width
        y = df['prob']
        lo = df['prob'] - df['asymp.LCL']
        hi = df['asymp.UCL'] - df['prob']
        ax.bar(x_pos, y, width=width, color=emotion_colors[code], label=full_label[code], zorder=3)
        ax.errorbar(x_pos, y, yerr=[lo, hi], fmt='none', capsize=3, color='black', zorder=4)
    ax.set_xticks(np.arange(len(x_levels)))
    ax.set_xticklabels(x_levels, rotation=15)
    ax.set_ylabel('Predicted Rejection Rate')
    ax.set_ylim(0, 1.05)
    ax.set_title(f'Rejection Rate: Interaction (Emotion × {analysis_type.capitalize()})')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(title='Emotion', loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(results_dir / "rejection_glmm_interaction_bar.tiff", dpi=220)
    plt.close(fig)

    # ========== 交互折线图 ==========
    fig, ax = plt.subplots(figsize=(7, 4))
    for code in emotion_order:
        df = (pred_glmm[pred_glmm['emotion'] == code]
              .set_index('x_norm').reindex(x_levels).reset_index())
        if df.empty: continue
        y = df['prob']
        lo = df['prob'] - df['asymp.LCL']
        hi = df['asymp.UCL'] - df['prob']
        ax.plot(np.arange(len(x_levels)), y,
                color=emotion_colors[code], marker='o', linewidth=2.2, markersize=6, label=full_label[code], zorder=3)
        ax.errorbar(np.arange(len(x_levels)), y, yerr=[lo, hi], fmt='none', capsize=3, color=emotion_colors[code], zorder=4)
    ax.set_xticks(np.arange(len(x_levels)))
    ax.set_xticklabels(x_levels, rotation=15)
    ax.set_ylabel('Predicted Rejection Rate')
    ax.set_ylim(0, 1.05)
    ax.set_title(f'Rejection Rate: Interaction (Emotion × {analysis_type.capitalize()})')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(title='Emotion', loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(results_dir / "rejection_glmm_interaction_line.tiff", dpi=220)
    plt.close(fig)

    # ========== 热力图 ==========
    pivot_reject = pred_glmm.pivot(index='emotion', columns='x_norm', values='prob').loc[emotion_order, x_levels]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(pivot_reject, annot=True, fmt=".2f", cmap="Reds",
                cbar_kws={'label': 'Predicted Rejection Rate'}, ax=ax)
    ax.set_yticklabels([full_label[e] for e in pivot_reject.index], rotation=0)
    ax.set_xlabel("Offer Type" if analysis_type == "type" else "Offer Ratio")
    ax.set_ylabel("Emotion")
    ax.set_title(f"Heatmap: Predicted Rejection (GLMM Interaction) [{analysis_type}]")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(results_dir / "rejection_heatmap.tiff", dpi=220)
    plt.close(fig)

    # ========== RT主效应barplot ==========
    emo_lmm = lmm_main[lmm_main['effect']=="Emotion"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7,4))
    for i, row in emo_lmm.iterrows():
        code = row['contrast'].split()[0]
        est  = row['estimate']
        err  = CI * row['SE']
        ax.bar(i, est, yerr=err,
               color=emotion_colors.get(code, "#cccccc"),
               edgecolor='black', capsize=4)
    ax.set_xticks(np.arange(len(emo_lmm)))
    ax.set_xticklabels([full_label.get(c.split()[0], c) for c in emo_lmm['contrast']], rotation=15)
    ax.set_ylabel('Log-RT Difference')
    ax.set_title(f'Reaction Time: Main Effects (Emotion) [{analysis_type}]')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(results_dir / "rt_lmm_main_emotion.tiff", dpi=220)
    plt.close(fig)

    # ========== RT交互barplot ==========
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.15
    for idx, code in enumerate(emotion_order):
        df = (pred_lmm[pred_lmm['emotion'] == code]
              .set_index('x_norm').reindex(x_levels).reset_index())
        if df.empty: continue
        x_pos = np.arange(len(x_levels)) + (idx - 2) * width
        y = df['RT_pred_ms']
        lo = df['RT_pred_ms'] - df['lower_ms']
        hi = df['upper_ms'] - df['RT_pred_ms']
        ax.bar(x_pos, y, width=width, color=emotion_colors[code], label=full_label[code], zorder=3)
        ax.errorbar(x_pos, y, yerr=[lo, hi], fmt='none', capsize=3, color='black', zorder=4)
    ax.set_xticks(np.arange(len(x_levels)))
    ax.set_xticklabels(x_levels, rotation=15)
    ax.set_ylabel('Predicted RT (ms)')
    ax.set_ylim(500, None)
    ax.set_title(f'Reaction Time: Interaction (Emotion × {analysis_type.capitalize()})')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(title='Emotion', loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(results_dir / "rt_lmm_interaction_bar.tiff", dpi=220)
    plt.close(fig)

    # ========== RT交互折线图 ==========
    fig, ax = plt.subplots(figsize=(7, 4))
    for code in emotion_order:
        df = (pred_lmm[pred_lmm['emotion'] == code]
              .set_index('x_norm').reindex(x_levels).reset_index())
        if df.empty: continue
        y = df['RT_pred_ms']
        lo = df['RT_pred_ms'] - df['lower_ms']
        hi = df['upper_ms'] - df['RT_pred_ms']
        ax.plot(np.arange(len(x_levels)), y,
                color=emotion_colors[code], marker='o', linewidth=2.2, markersize=6, label=full_label[code], zorder=3)
        ax.errorbar(np.arange(len(x_levels)), y, yerr=[lo, hi], fmt='none', capsize=3, color=emotion_colors[code], zorder=4)
    ax.set_xticks(np.arange(len(x_levels)))
    ax.set_xticklabels(x_levels, rotation=15)
    ax.set_ylabel('Predicted RT (ms)')
    ax.set_ylim(500, None)
    ax.set_title(f'Reaction Time: Interaction (Emotion × {analysis_type.capitalize()})')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(title='Emotion', loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(results_dir / "rt_lmm_interaction_line.tiff", dpi=220)
    plt.close(fig)

    # ========== RT热力图 ==========
    pivot_rt = pred_lmm.pivot(index='emotion', columns='x_norm', values='RT_pred_ms').loc[emotion_order, x_levels]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(pivot_rt, annot=True, fmt=".0f", cmap="Blues",
                cbar_kws={'label': 'Predicted RT (ms)'}, ax=ax)
    ax.set_yticklabels([full_label[e] for e in pivot_rt.index], rotation=0)
    ax.set_xlabel("Offer Type" if analysis_type == "type" else "Offer Ratio")
    ax.set_ylabel("Emotion")
    ax.set_title(f"Heatmap: Predicted RT (LMM Interaction) [{analysis_type}]")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(results_dir / "rt_heatmap.tiff", dpi=220)
    plt.close(fig)

    # ========== 小提琴图：拒绝率+RT ==========
    violin_file = results_dir / f"rejection_rate_by_subject.csv"
    if not violin_file.exists():
        violin_file = results_dir / f"acceptance_rate_by_subject.csv"  # 向后兼容
    if violin_file.exists():
        accept_df = pd.read_csv(violin_file)
        if analysis_type == "ratio":
            facet_col = "offer_ratio"
            facet_vals = [x for x in ratio_levels if x in accept_df[facet_col].unique()]
        else:
            facet_col = "offer_type"
            facet_vals = [x for x in type_levels if x in accept_df[facet_col].unique()]
        for val in facet_vals:
            sub = accept_df[accept_df[facet_col] == val]
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            sns.violinplot(
                data=sub, x="emotion", y="rejection_rate",
                palette=custom_palette, linewidth=0, alpha=1, cut=0, inner=None, ax=ax
            )
            sns.stripplot(
                data=sub, x="emotion", y="rejection_rate",
                color="black", jitter=0.23, size=4, alpha=0.65, ax=ax, zorder=5
            )
            means = sub.groupby("emotion")["rejection_rate"].mean().reindex(emotion_order)
            ns = sub.groupby("emotion")["rejection_rate"].count().reindex(emotion_order)
            ses = sub.groupby("emotion")["rejection_rate"].sem().reindex(emotion_order)
            from scipy.stats import t
            cis = ses * t.ppf(0.975, ns-1)
            ax.errorbar(
                x=np.arange(len(emotion_order)),
                y=means, yerr=cis,
                fmt='o', color="black", markersize=6.2, capsize=4.1, lw=1.2, zorder=9
            )
            for i, emo in enumerate(emotion_order):
                if not np.isnan(means[emo]):
                    ax.plot(i, means[emo], marker="o", markersize=8, color=edge_palette[i], zorder=10, alpha=0.89)
            ax.set_xticks(np.arange(len(emotion_order)))
            ax.set_xticklabels([full_label[e] for e in emotion_order], rotation=13)
            ax.set_ylim(0, 1.07)
            ax.set_ylabel("Rejection Rate")
            ax.set_title(f"Rejection Rate by Emotion\n{facet_col.replace('_',' ').title()}: {val}")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            safe_val = str(val).replace(":", "_")
            plt.savefig(results_dir / f"violin_rejection_{facet_col}_{safe_val}.tiff", dpi=220)
            plt.close(fig)
    # RT violin
    meanrt_file = results_dir / f"mean_rt_by_subject.csv"
    if meanrt_file.exists():
        meanrt_df = pd.read_csv(meanrt_file)
        if analysis_type == "ratio":
            facet_col = "offer_ratio"
            facet_vals = [x for x in ratio_levels if x in meanrt_df[facet_col].unique()]
        else:
            facet_col = "offer_type"
            facet_vals = [x for x in type_levels if x in meanrt_df[facet_col].unique()]
        for val in facet_vals:
            sub = meanrt_df[meanrt_df[facet_col] == val]
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            sns.violinplot(
                data=sub, x="emotion", y="mean_rt",
                palette=custom_palette, linewidth=0, alpha=1, cut=0, inner=None, ax=ax
            )
            sns.stripplot(
                data=sub, x="emotion", y="mean_rt",
                color="black", jitter=0.23, size=4, alpha=0.65, ax=ax, zorder=5
            )
            means = sub.groupby("emotion")["mean_rt"].mean().reindex(emotion_order)
            ns = sub.groupby("emotion")["mean_rt"].count().reindex(emotion_order)
            ses = sub.groupby("emotion")["mean_rt"].sem().reindex(emotion_order)
            from scipy.stats import t
            cis = ses * t.ppf(0.975, ns-1)
            ax.errorbar(
                x=np.arange(len(emotion_order)),
                y=means, yerr=cis,
                fmt='o', color="black", markersize=6.2, capsize=4.1, lw=1.2, zorder=9
            )
            for i, emo in enumerate(emotion_order):
                if not np.isnan(means[emo]):
                    ax.plot(i, means[emo], marker="o", markersize=8, color=edge_palette[i], zorder=10, alpha=0.89)
            ax.set_xticks(np.arange(len(emotion_order)))
            ax.set_xticklabels([full_label[e] for e in emotion_order], rotation=13)
            ax.set_ylim(500, np.nanmax(meanrt_df["mean_rt"]) * 1.13)
            ax.set_ylabel("Mean RT (ms)")
            ax.set_title(f"Mean RT by Emotion\n{facet_col.replace('_',' ').title()}: {val}")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            safe_val = str(val).replace(":", "_")
            plt.savefig(results_dir / f"violin_rt_{facet_col}_{safe_val}.tiff", dpi=220)
            plt.close(fig)

print("\n全部.tiff格式的可视化图片已自动输出至 analysis_output/results_refuse/ratio 和 type 下！")

# ============================================
# Part 3: Unfair条件下拒绝反应的RT可视化
# ============================================

print("\n=== Part 3: Unfair条件拒绝反应RT可视化 ===")

# 设置Part 3的目录
unfair_dir = results_root / "unfair_reject"

# 检查目录和文件是否存在
if unfair_dir.exists():
    required_files = [
        "RT_descriptives.csv",
        "RT_emmeans.csv",
        "RT_effect_sizes.csv",
        "RT_by_subject.csv"
    ]
    
    if all((unfair_dir / f).exists() for f in required_files):
        
        # 读取数据
        desc_stats = pd.read_csv(unfair_dir / "RT_descriptives.csv")
        emmeans = pd.read_csv(unfair_dir / "RT_emmeans.csv")
        effect_sizes = pd.read_csv(unfair_dir / "RT_effect_sizes.csv")
        by_subject = pd.read_csv(unfair_dir / "RT_by_subject.csv")
        
        # 确保情绪顺序正确
        emmeans['emotion'] = pd.Categorical(emmeans['emotion'], categories=emotion_order, ordered=True)
        emmeans = emmeans.sort_values('emotion')
        
        # ========== 图1: 主效应条形图 ==========
        fig, ax = plt.subplots(figsize=(7, 5))
        
        x_pos = np.arange(len(emmeans))
        colors_list = [emotion_colors[e] for e in emmeans['emotion']]
        
        bars = ax.bar(x_pos, emmeans['RT_pred_ms'], 
                      color=colors_list, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # 添加误差线
        errors = [(emmeans['RT_pred_ms'] - emmeans['lower_CI']).values,
                  (emmeans['upper_CI'] - emmeans['RT_pred_ms']).values]
        ax.errorbar(x_pos, emmeans['RT_pred_ms'], yerr=errors,
                    fmt='none', color='black', capsize=5, linewidth=1.5)
        
        # 在条形上方添加数值
        for i, (val, bar) in enumerate(zip(emmeans['RT_pred_ms'], bars)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([full_label[e] for e in emmeans['emotion']], rotation=0)
        ax.set_ylabel('Predicted RT (ms)', fontsize=12)
        ax.set_ylim(800, 1100)
        ax.set_title('Unfair Rejection RT: Model Predictions with 95% CI', fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(unfair_dir / "RT_bars.tiff", dpi=220)
        plt.close(fig)
        
        # ========== 图2: 小提琴图（个体数据）==========
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # 准备数据
        by_subject['emotion'] = pd.Categorical(by_subject['emotion'], categories=emotion_order, ordered=True)
        by_subject_sorted = by_subject.sort_values('emotion')
        
        # 绘制小提琴图
        violin_data = []
        for emo in emotion_order:
            data = by_subject_sorted[by_subject_sorted['emotion']==emo]['mean_RT'].values
            if len(data) > 0:
                violin_data.append(data)
            else:
                violin_data.append([])
        
        parts = ax.violinplot(
            [d for d in violin_data if len(d) > 0],
            positions=[i for i, d in enumerate(violin_data) if len(d) > 0],
            widths=0.7,
            showmeans=False,
            showextrema=False
        )
        
        # 设置小提琴颜色
        for i, pc in enumerate(parts['bodies']):
            valid_idx = [j for j, d in enumerate(violin_data) if len(d) > 0][i]
            pc.set_facecolor(emotion_colors[emotion_order[valid_idx]])
            pc.set_alpha(0.4)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)
        
        # 添加个体数据点
        for i, emo in enumerate(emotion_order):
            y = by_subject_sorted[by_subject_sorted['emotion']==emo]['mean_RT'].values
            if len(y) > 0:
                x = np.random.normal(i, 0.08, size=len(y))
                ax.scatter(x, y, alpha=0.6, s=20, color='gray', zorder=3)
        
        # 添加均值和误差线
        means = by_subject_sorted.groupby('emotion')['mean_RT'].mean().reindex(emotion_order)
        sems = by_subject_sorted.groupby('emotion')['mean_RT'].sem().reindex(emotion_order)
        
        for i, emo in enumerate(emotion_order):
            if not pd.isna(means[emo]):
                ax.scatter(i, means[emo], color='black', s=100, zorder=5, marker='D')
                ax.errorbar(i, means[emo], yerr=1.96*sems[emo],
                           fmt='none', color='black', capsize=5, linewidth=2, zorder=4)
        
        # 添加模型预测值（红色菱形）
        for i, emo in enumerate(emotion_order):
            if emo in emmeans['emotion'].values:
                pred_val = emmeans[emmeans['emotion']==emo]['RT_pred_ms'].values[0]
                ax.scatter(i, pred_val, color='red', s=120, zorder=6, marker='D', 
                          edgecolors='darkred', linewidth=1.5)
        
        ax.set_xticks(np.arange(len(emotion_order)))
        ax.set_xticklabels([full_label[e] for e in emotion_order])
        ax.set_ylabel('Mean RT (ms)', fontsize=12)
        ax.set_ylim(600, 1400)
        ax.set_title('Individual Participant Data: Unfair Rejection RT\nBlack: observed mean ± 95% CI | Red: model prediction', 
                     fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='D', color='w', markerfacecolor='black', markersize=8, label='Observed mean'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='red', markersize=8, label='Model prediction')
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=False)
        
        plt.tight_layout()
        plt.savefig(unfair_dir / "RT_violin.tiff", dpi=220)
        plt.close(fig)
        
        # ========== 图3: 效应量森林图 ==========
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 筛选显著效应
        sig_effects = effect_sizes[effect_sizes['p.value'] < 0.05].copy()
        
        if len(sig_effects) > 0:
            sig_effects = sig_effects.sort_values('cohen_d')
            y_pos = np.arange(len(sig_effects))
            
            # 根据p值设置颜色
            colors_eff = []
            for p in sig_effects['p.value']:
                if p < 0.001:
                    colors_eff.append('#d73027')  # 深红
                elif p < 0.01:
                    colors_eff.append('#fc8d59')  # 橙色
                else:
                    colors_eff.append('#fee090')  # 浅黄
            
            ax.barh(y_pos, sig_effects['cohen_d'], 
                    color=colors_eff, edgecolor='black', linewidth=1, alpha=0.8)
            
            # 添加误差线（如果有SE）
            if 'SE' in sig_effects.columns:
                errors = 1.96 * sig_effects['SE'] / 0.2729  # 使用模型的残差标准差
                ax.errorbar(sig_effects['cohen_d'], y_pos, xerr=errors,
                           fmt='none', color='black', capsize=3, linewidth=1)
            
            # 添加参考线
            ax.axvline(0, color='gray', linestyle='-', linewidth=1)
            ax.axvline(-0.2, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.axvline(-0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            
            # 添加标签
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sig_effects['contrast'])
            ax.set_xlabel("Cohen's d", fontsize=12)
            ax.set_title("Effect Sizes for Significant Contrasts (FDR < 0.05)\nNegative = faster RT", 
                        fontsize=12, fontweight='bold')
            
            # 添加效应量分类文字
            ax.text(-0.35, ax.get_ylim()[1] * 0.95, 'Small', fontsize=9, alpha=0.7)
            ax.text(-0.65, ax.get_ylim()[1] * 0.95, 'Medium', fontsize=9, alpha=0.7)
            
            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#d73027', edgecolor='black', label='p < 0.001'),
                Patch(facecolor='#fc8d59', edgecolor='black', label='p < 0.01'),
                Patch(facecolor='#fee090', edgecolor='black', label='p < 0.05')
            ]
            ax.legend(handles=legend_elements, loc='lower right', frameon=False)
        else:
            ax.text(0.5, 0.5, 'No significant effects (FDR < 0.05)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(unfair_dir / "RT_forest.tiff", dpi=220)
        plt.close(fig)
        
        # ========== 图4: 效应量热力图 ==========
        # 创建效应量矩阵
        emotions_in_order = emotion_order
        effect_matrix = np.zeros((5, 5))
        
        for _, row in effect_sizes.iterrows():
            contrast = row['contrast']
            if ' - ' in contrast:
                emo1, emo2 = contrast.split(' - ')
                if emo1 in emotions_in_order and emo2 in emotions_in_order:
                    i = emotions_in_order.index(emo1)
                    j = emotions_in_order.index(emo2)
                    effect_matrix[i, j] = row['cohen_d']
                    effect_matrix[j, i] = -row['cohen_d']
        
        # 创建mask（对角线）
        mask = np.eye(5, dtype=bool)
        
        fig, ax = plt.subplots(figsize=(7, 6))
        
        sns.heatmap(effect_matrix, 
                   mask=mask,
                   annot=True, 
                   fmt='.2f',
                   cmap='RdBu_r',
                   center=0,
                   vmin=-0.5, vmax=0.5,
                   square=True,
                   linewidths=1,
                   cbar_kws={'label': "Cohen's d", 'shrink': 0.8},
                   ax=ax)
        
        ax.set_xticklabels([full_label[e] for e in emotions_in_order], rotation=45, ha='right')
        ax.set_yticklabels([full_label[e] for e in emotions_in_order], rotation=0)
        ax.set_title("Pairwise Effect Sizes: Unfair Rejection RT\n(Row - Column)", 
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(unfair_dir / "RT_heatmap.tiff", dpi=220)
        plt.close(fig)
        
        print("Part 3 可视化完成！图片保存在 analysis_output/results_refuse/unfair_reject/")
        
    else:
        print("警告：Part 3 某些必要文件缺失，跳过可视化")
        for f in required_files:
            if not (unfair_dir / f).exists():
                print(f"  缺失文件: {f}")
else:
    print("警告：unfair_reject 文件夹不存在，跳过Part 3可视化")

print("\n所有可视化完成！")
