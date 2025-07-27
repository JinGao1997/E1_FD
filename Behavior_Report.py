from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ===================== 配置 =====================
analysis_types = ["ratio", "type"]  # 两种分析类型，分别画两套
for analysis_type in analysis_types:
    print(f"\n=== 正在绘制 {analysis_type} 分析结果 ===")

    # 自动定位到当前脚本同级目录下的 results/type 或 results/ratio
    base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path().cwd()
    results_dir = base_dir / "analysis_output" / "results" / analysis_type

    # ======= 读取数据 =======
    glmm_main = pd.read_csv(results_dir / "glmm_main_effects.csv")
    pred_glmm = pd.read_csv(results_dir / "glmm_interaction_preds.csv")
    lmm_main  = pd.read_csv(results_dir / "lmm_main_effects.csv")
    pred_lmm  = pd.read_csv(results_dir / "lmm_interaction_preds.csv")

    # 判别自变量类型
    if analysis_type == "ratio":
        ratio_col = "offer_ratio"
        ratio_levels = [r for r in ["5:5", "4:6", "3:7", "2:8", "1:9"] if r in pred_glmm['offer_ratio'].unique()]
        type_col = None
    else:  # type
        ratio_col = "offer_type"
        ratio_levels = [r for r in ["fair", "unfair"] if r in pred_glmm['offer_type'].unique()]
        type_col = "offer_type"

    # ======= Ratio 标准化列（如有必要）=======
    def normalize_ratio(s):
        # 只标准化 ratio 字符串
        if ':' in s:
            a, b = s.split(':')
            return f"{int(a)}:{int(b)}"
        else:
            return s
    for df in (pred_glmm, pred_lmm):
        df['ratio_norm'] = df[ratio_col].apply(normalize_ratio)

    # ======= 缩写→全称 & 色盲友好配色 =======
    full_label = {
        'dis': 'disgust',
        'dom': 'dominance',
        'neu': 'neutral',
        'aff': 'affiliative',
        'enj': 'reward'
    }
    emotion_colors = {
        'dis': "#755627",    # 棕色（disgust）
        'dom': "#F5900C",    # 亮橘色（dominance）
        'neu': "#C5C5C5EC",  # 浅灰色（neutral）
        'aff': "#39E04F",    # 亮绿色（affiliation）
        'enj': "#FC0000"     # 亮红色（reward）
    }
    linestyles = {k: 'solid' for k in full_label}
    markers    = {'dis':'o','dom':'s','neu':'^','aff':'D','enj':'v'}
    linewidths = {'dis':2.5,'dom':2.0,'neu':2.0,'aff':2.5,'enj':2.0}
    sns.set(style="white", context="talk")
    plt.rcParams.update({
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'figure.dpi': 120,
    })
    x = np.arange(len(ratio_levels))
    CI = 1.96

    # ======= 1. GLMM 主效应 =======
    emo_glmm = glmm_main[glmm_main['effect']=="Emotion"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7,4))
    for i,row in emo_glmm.iterrows():
        code = row['contrast'].split()[0]
        or_  = row['odds.ratio']
        err  = CI * row['SE']
        ax.bar(i, or_, yerr=err,
               color=emotion_colors[code],
               edgecolor='black', capsize=4)
    ax.axhline(1, linestyle='--', color='gray')
    ax.set_xticks(np.arange(len(emo_glmm)))
    ax.set_xticklabels([full_label.get(c.split()[0], c) for c in emo_glmm['contrast']], rotation=15)
    ax.set_ylabel('Odds Ratio')
    ax.set_title(f'Acceptance Rates: Main Effects (Emotion) [{analysis_type}]')
    plt.tight_layout()
    plt.show()

    # ======= 2. GLMM 交互 =======
    fig, ax = plt.subplots(figsize=(7,4))
    for code in full_label:
        df = (pred_glmm[pred_glmm['emotion']==code]
              .set_index('ratio_norm').loc[ratio_levels].reset_index())
        y = df['prob']
        lo = df['prob'] - df['asymp.LCL']
        hi = df['asymp.UCL'] - df['prob']
        ax.plot(x, y,
                color=emotion_colors[code],
                linestyle=linestyles[code],
                marker=markers[code],
                linewidth=linewidths[code],
                markersize=6,
                label=full_label[code])
        ax.errorbar(x, y, yerr=[lo,hi], fmt='none', capsize=3, color=emotion_colors[code])
    ax.set_xticks(x)
    ax.set_xticklabels(ratio_levels, rotation=15)
    ax.set_ylabel('Predicted Acceptance')
    ax.set_title(f'Acceptance Rates: Interaction (Emotion × {analysis_type.capitalize()})')
    ax.legend(title='Emotion', loc='upper left', bbox_to_anchor=(1.02,1))
    plt.tight_layout()
    plt.show()

    # ======= 3. LMM 主效应 =======
    emo_lmm = lmm_main[lmm_main['effect']=="Emotion"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7,4))
    for i,row in emo_lmm.iterrows():
        code = row['contrast'].split()[0]
        est  = row['estimate']
        err  = CI * row['SE']
        ax.bar(i, est, yerr=err,
               color=emotion_colors[code],
               edgecolor='black', capsize=4)
    ax.set_xticks(np.arange(len(emo_lmm)))
    ax.set_xticklabels([full_label.get(c.split()[0], c) for c in emo_lmm['contrast']], rotation=15)
    ax.set_ylabel('Log-RT Difference')
    ax.set_title(f'Reaction Time: Main Effects (Emotion) [{analysis_type}]')
    plt.tight_layout()
    plt.show()

    # ======= 4. LMM 交互 =======
    fig, ax = plt.subplots(figsize=(7,4))
    for code in full_label:
        df = (pred_lmm[pred_lmm['emotion']==code]
              .set_index('ratio_norm').loc[ratio_levels].reset_index())
        y = df['RT_pred_ms']
        lo = df['RT_pred_ms'] - df['lower_ms']
        hi = df['upper_ms']   - df['RT_pred_ms']
        ax.plot(x, y,
                color=emotion_colors[code],
                linestyle=linestyles[code],
                marker=markers[code],
                linewidth=linewidths[code],
                markersize=6,
                label=full_label[code])
        ax.errorbar(x, y, yerr=[lo,hi], fmt='none', capsize=3, color=emotion_colors[code])
    ax.set_xticks(x)
    ax.set_xticklabels(ratio_levels, rotation=15)
    ax.set_ylabel('Predicted RT (ms)')
    ax.set_title(f'Reaction Time: Interaction (Emotion × {analysis_type.capitalize()})')
    ax.legend(title='Emotion', loc='upper left', bbox_to_anchor=(1.02,1))
    plt.tight_layout()
    plt.show()

    # ======= 5. 热力图：Predicted Acceptance =======
    pivot_accept = pred_glmm.pivot(
        index='emotion', columns='ratio_norm', values='prob'
    ).loc[full_label.keys(), ratio_levels]  # 行列顺序
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(
        pivot_accept,
        annot=True, fmt=".2f",
        cmap="Reds",
        cbar_kws={'label': 'Predicted Acceptance'},
        ax=ax
    )
    ax.set_yticklabels([full_label[e] for e in pivot_accept.index], rotation=0)
    ax.set_xlabel("Offer Type" if analysis_type == "type" else "Offer Ratio")
    ax.set_ylabel("Emotion")
    ax.set_title(f"Heatmap: Predicted Acceptance (GLMM Interaction) [{analysis_type}]")
    plt.tight_layout()
    plt.show()

    # ======= 6. 热力图：Predicted RT =======
    pivot_rt = pred_lmm.pivot(
        index='emotion', columns='ratio_norm', values='RT_pred_ms'
    ).loc[full_label.keys(), ratio_levels]
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(
        pivot_rt,
        annot=True, fmt=".0f",
        cmap="Blues",
        cbar_kws={'label': 'Predicted RT (ms)'},
        ax=ax
    )
    ax.set_yticklabels([full_label[e] for e in pivot_rt.index], rotation=0)
    ax.set_xlabel("Offer Type" if analysis_type == "type" else "Offer Ratio")
    ax.set_ylabel("Emotion")
    ax.set_title(f"Heatmap: Predicted RT (LMM Interaction) [{analysis_type}]")
    plt.tight_layout()
    plt.show()




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import t

# ========== 1. 标签与色板 ==========
full_label = {
    'dis': 'Disgust',
    'dom': 'Dominance',
    'neu': 'Neutral',
    'aff': 'Affiliative',
    'enj': 'Reward'
}
emotion_order = ["dis", "dom", "neu", "aff", "enj"]
emotion_colors = {
    'dis': "#8c510a",    # 深棕
    'dom': "#e08214",    # 亮橙
    'neu': "#bababa",    # 灰
    'aff': "#35978f",    # 青绿
    'enj': "#d73027"     # 亮红
}
def rgba(hex_color, alpha=0.36):
    import matplotlib.colors as mcolors
    rgb = mcolors.hex2color(hex_color)
    return tuple(list(rgb) + [alpha])
custom_palette = [rgba(emotion_colors[e], 0.36) for e in emotion_order]
edge_palette = [emotion_colors[e] for e in emotion_order]

# ========== 2. 数据读取与基本变量 ==========
DATA_PATH = "E:/PD_E1_UG_jg/EEG_R_Python_Pipeline_JG_Backup/Offers__Behavior/BehaviorDataAnalysis/trials.csv"
df_raw = pd.read_csv(DATA_PATH, encoding="utf-8")
df_raw["offer_ratio"] = pd.Categorical(
    df_raw["Offers_You"].astype(str) + ":" + df_raw["Offers_Other"].astype(str),
    categories=["5:5", "4:6", "3:7", "2:8", "1:9"], ordered=True
)
df_raw["offer_type"] = pd.Categorical(
    np.select(
        [
            df_raw["offer_ratio"].isin(["5:5", "4:6"]),
            df_raw["offer_ratio"].isin(["2:8", "1:9"])
        ],
        ["fair", "unfair"], default="other"
    ),
    categories=["fair", "unfair", "other"], ordered=True
)
df_raw["reaction_bin"] = (df_raw["reaction"] == 1).astype(int)
df_raw["emotion"] = pd.Categorical(
    df_raw["emotion"], categories=emotion_order, ordered=True
)
df_raw["participant_id"] = df_raw["participant_id"].astype(str)
out_ratio = Path("analysis_output/results/ratio"); out_ratio.mkdir(parents=True, exist_ok=True)
out_type  = Path("analysis_output/results/type");  out_type.mkdir(parents=True, exist_ok=True)

# ========== 3. 各类统计表：接受率与平均RT ==========
# Ratio分析
df_ratio = df_raw[(df_raw["RT"]>=200) & (df_raw["RT"]<=3000) & (~df_raw["reaction"].isna()) & (~df_raw["offer_ratio"].isna())].copy()
accept_ratio = (
    df_ratio.groupby(["participant_id", "emotion", "offer_ratio"], observed=True)
    .agg(acceptance_rate=("reaction_bin", "mean"), n_trials=("reaction_bin", "count"))
    .reset_index()
)
accept_ratio["emotion_label"] = accept_ratio["emotion"].map(full_label)
accept_ratio.to_csv(out_ratio / "acceptance_rate_by_subject.csv", index=False)
meanrt_ratio = (
    df_ratio.groupby(["participant_id", "emotion", "offer_ratio"], observed=True)
    .agg(mean_rt=("RT", "mean"), n_trials=("RT", "count"))
    .reset_index()
)
meanrt_ratio["emotion_label"] = meanrt_ratio["emotion"].map(full_label)
meanrt_ratio.to_csv(out_ratio / "mean_rt_by_subject.csv", index=False)

# Type分析
df_type = df_raw[
    (df_raw["RT"]>=200) & (df_raw["RT"]<=3000) & (~df_raw["reaction"].isna()) &
    (df_raw["offer_type"].isin(["fair", "unfair"]))
].copy()
accept_type = (
    df_type.groupby(["participant_id", "emotion", "offer_type"], observed=True)
    .agg(acceptance_rate=("reaction_bin", "mean"), n_trials=("reaction_bin", "count"))
    .reset_index()
)
accept_type["emotion_label"] = accept_type["emotion"].map(full_label)
accept_type.to_csv(out_type / "acceptance_rate_by_subject.csv", index=False)
meanrt_type = (
    df_type.groupby(["participant_id", "emotion", "offer_type"], observed=True)
    .agg(mean_rt=("RT", "mean"), n_trials=("RT", "count"))
    .reset_index()
)
meanrt_type["emotion_label"] = meanrt_type["emotion"].map(full_label)
meanrt_type.to_csv(out_type / "mean_rt_by_subject.csv", index=False)

# ========== 4. 主绘图函数 ==========
def violin_scatter_summary_nice(
    df, x, y, facet, order, palette, edge_palette, ylabel, title, filename, ylims=None
):
    facet_cats = df[facet].dropna().unique().tolist()
    n_col = 3
    n_row = int(np.ceil(len(facet_cats)/n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*4.2, n_row*4.2), sharey=True)
    axes = np.atleast_2d(axes)
    for idx, facet_value in enumerate(facet_cats):
        r, c = divmod(idx, n_col)
        ax = axes[r][c]
        sub = df[df[facet] == facet_value]
        # 小提琴
        sns.violinplot(
            data=sub, x=x, y=y, order=order, ax=ax,
            linewidth=0, palette=palette, alpha=1, width=0.96, cut=0, inner=None
        )
        # 散点
        sns.stripplot(
            data=sub, x=x, y=y, order=order, ax=ax,
            color="black", jitter=0.25, size=4.1, alpha=0.67, zorder=5
        )
        # 均值和95%CI
        means = sub.groupby(x)[y].mean()
        ns = sub.groupby(x)[y].count()
        ses = sub.groupby(x)[y].sem()
        cis = ses * t.ppf(0.975, ns-1)
        ax.errorbar(
            x=np.arange(len(order)),
            y=means[order], yerr=cis[order],
            fmt='o', color="black", markersize=6.5, capsize=4.1, lw=1.4, zorder=9
        )
        # 大色点覆盖
        for i, emo in enumerate(order):
            if emo in means.index:
                ax.plot(i, means[emo], marker="o", markersize=8.3, color=edge_palette[i], zorder=10, alpha=0.89)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([full_label[e] for e in emotion_order], rotation=13)
        if ylims is not None:
            ax.set_ylim(*ylims)
        ax.set_title(f"{facet.replace('_',' ').capitalize()}: {facet_value}", fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.grid(False)
    for j in range(idx+1, n_row*n_col):
        r, c = divmod(j, n_col)
        axes[r][c].set_visible(False)
    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.96)
    plt.tight_layout(rect=[0,0,1,0.93])
    plt.savefig(filename, dpi=220)
    plt.show()

# ========== 5. 可视化 - 接受率 ==========
violin_scatter_summary_nice(
    accept_ratio, x="emotion", y="acceptance_rate", facet="offer_ratio",
    order=emotion_order, palette=custom_palette, edge_palette=edge_palette,
    ylabel="Acceptance Rate",
    title="Acceptance Rate by Emotion and Offer Ratio",
    filename=str(out_ratio/"acceptance_violin_ratio_nice.png"),
    ylims=(0, 1.07)
)
violin_scatter_summary_nice(
    accept_type, x="emotion", y="acceptance_rate", facet="offer_type",
    order=emotion_order, palette=custom_palette, edge_palette=edge_palette,
    ylabel="Acceptance Rate",
    title="Acceptance Rate by Emotion and Offer Type",
    filename=str(out_type/"acceptance_violin_type_nice.png"),
    ylims=(0, 1.07)
)

# ========== 6. 可视化 - RT分布 ==========
violin_scatter_summary_nice(
    meanrt_ratio, x="emotion", y="mean_rt", facet="offer_ratio",
    order=emotion_order, palette=custom_palette, edge_palette=edge_palette,
    ylabel="Mean Reaction Time (ms)",
    title="Mean RT by Emotion and Offer Ratio",
    filename=str(out_ratio/"rt_violin_ratio_nice.png"),
    ylims=(200, np.nanmax(meanrt_ratio["mean_rt"])*1.13)
)
violin_scatter_summary_nice(
    meanrt_type, x="emotion", y="mean_rt", facet="offer_type",
    order=emotion_order, palette=custom_palette, edge_palette=edge_palette,
    ylabel="Mean Reaction Time (ms)",
    title="Mean RT by Emotion and Offer Type",
    filename=str(out_type/"rt_violin_type_nice.png"),
    ylims=(200, np.nanmax(meanrt_type["mean_rt"])*1.13)
)


