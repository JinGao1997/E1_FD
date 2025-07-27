import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import plotly.express as px

# ========== 1. 数据读取与整理 ==========
data_path = r"C:\EEG_R_Python_Pipeline_JG\E1_UG\export\offer_phase\trials.csv"  # 修改为你的路径
df = pd.read_csv(data_path)
x_order = ["5:5", "6:4", "8:2", "9:1"]

def precise_offer_type(row):
    offer = f"{row['Offers_Other']}:{row['Offers_You']}"
    if offer in x_order:
        return offer
    else:
        return None

df['offer_type2'] = df.apply(precise_offer_type, axis=1)
df['offer_label'] = df['Offers_Other'].astype(str) + ":" + df['Offers_You'].astype(str)

# 统计每个被试对每个 offer 的接受率
pivot = (
    df[df['offer_type2'].notnull()]
    .groupby(['participant_id', 'offer_type2'])
    .reaction
    .apply(lambda x: (x == 1).mean())
    .unstack()
    .reset_index()
)
pivot.to_csv("accept_rate_by_offer.csv", index=False)

# 供可视化用的长格式
plot_df = pivot.melt(id_vars="participant_id", var_name="offer_type", value_name="accept_rate")
plot_df['pid_num'] = plot_df['participant_id'].apply(lambda x: x[-2:] if len(x) > 2 else x)

# ========== 2. K-means聚类 ==========
# 用这4种offer的接受率聚类
offer_accepts = pivot[x_order].fillna(0).values
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(offer_accepts)
pid2cluster = dict(zip(pivot['participant_id'], cluster_labels))
plot_df['cluster'] = plot_df['participant_id'].map(pid2cluster)
palette = sns.color_palette("Set2", n_clusters)

# ========== 3. matplotlib静态可视化 ==========
plt.figure(figsize=(13, 6))
for pid, subdf in plot_df.groupby('participant_id'):
    xs = [x_order.index(x) for x in subdf['offer_type']]
    ys = subdf['accept_rate'].values
    cidx = pid2cluster[pid]
    # jitter防止线完全重叠
    jitter = np.random.normal(0, 0.04, size=len(xs))
    plt.plot(np.array(xs) + jitter, ys, color=palette[cidx], alpha=0.5, linewidth=1.2, zorder=1)
    plt.scatter(np.array(xs) + jitter, ys, color=palette[cidx], alpha=0.7, s=30, zorder=2)
# 箱线图（透明背景）
sns.boxplot(data=plot_df, x="offer_type", y="accept_rate", showcaps=True, boxprops={'facecolor':'none', 'zorder':3},
            showfliers=False, order=x_order)
plt.ylabel("Acceptance Rate")
plt.xlabel("Offer Type")
plt.title(f"Acceptance Rate per Offer Type per Participant\nColored by KMeans Cluster (n={n_clusters})")
plt.ylim(-0.05, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.4)
# 图例
for idx, color in enumerate(palette):
    plt.plot([], [], color=color, label=f"Cluster {idx+1}")
plt.legend(loc="lower left", bbox_to_anchor=(1, 0.1))
plt.tight_layout()
plt.show()

# ========== 4. Plotly交互式折线图 ==========
plot_df['offer_type'] = pd.Categorical(plot_df['offer_type'], categories=x_order, ordered=True)
fig = px.line(
    plot_df,
    x="offer_type", y="accept_rate", color="cluster",
    line_group="participant_id", hover_name="participant_id",
    markers=True,
    color_discrete_sequence=px.colors.qualitative.Set2,
    category_orders={"offer_type": x_order}
)
fig.update_traces(line=dict(width=1), marker=dict(size=8, opacity=0.8))
fig.update_layout(
    title="Acceptance Rate per Offer Type (Each Line = 1 Participant, Clustered)",
    yaxis=dict(range=[-0.05, 1.05]),
    xaxis_title="Offer Type",
    yaxis_title="Acceptance Rate",
    legend_title="Cluster",
    template="simple_white"
)
fig.show()
