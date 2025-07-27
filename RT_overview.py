import pandas as pd
import glob
import os

# 路径设置
input_dir = r"C:\EEG_R_Python_Pipeline_JG\E1_UG\data\csv_filtered"
output_dir = r"C:\EEG_R_Python_Pipeline_JG\E1_UG\data\csv_filtered_clean"
os.makedirs(output_dir, exist_ok=True)

csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

count_extreme = 0
detail_list = []

for file in csv_files:
    df = pd.read_csv(file, sep=None, engine='python')
    # 检查RT<150ms 或 RT>=3000ms的trial
    extreme_trials = df[(df['RT'] < 150) | (df['RT'] >= 3000)].copy()
    n = len(extreme_trials)
    if n > 0:
        extreme_trials['source_file'] = os.path.basename(file)
        subject = extreme_trials['subject'].iloc[0] if 'subject' in extreme_trials.columns else os.path.splitext(os.path.basename(file))[0]
        print(f"被试 {subject} 文件 {os.path.basename(file)} 中有 {n} 个极端RT（<150ms或>=3000ms）的trial")
        detail_list.append(extreme_trials)
        count_extreme += n
    # 剔除极端trial后保存
    df_clean = df[(df['RT'] >= 150) & (df['RT'] < 3000)].copy()
    base = os.path.basename(file)
    out_file = os.path.join(output_dir, base)
    df_clean.to_csv(out_file, index=False)
    print(f"{base} 已保存clean版，剩余trial数: {len(df_clean)}")

# 合并极端trial导出
if detail_list:
    all_extreme = pd.concat(detail_list, ignore_index=True)
    all_extreme.to_csv("RT_OfferPhase_lt150ms_or_ge3000ms_trials.csv", index=False)
    print("\n详细信息已导出到 RT_lt150ms_or_ge3000ms_trials.csv")
else:
    print("未检测到RT<150ms或RT>=3000ms的trial。")

print(f"\n所有被试总共 {count_extreme} 个RT<150ms或RT>=3000ms的trial，clean数据全部已保存。")
