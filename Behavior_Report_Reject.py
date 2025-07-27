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
