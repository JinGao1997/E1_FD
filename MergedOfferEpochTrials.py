import os
import pandas as pd
import re
from tqdm import tqdm

# ========== 配置参数 ==========
root_dir = r"E:\PD_E1_UG_jg\EEG_R_Python_Pipeline_JG_Backup\E1_UG\epochs"
phase_folder = root_dir
cleaned_folder = os.path.join(root_dir, "offer_phase", "cleaned")
os.makedirs(cleaned_folder, exist_ok=True)

meta_cols = [
    'participant_id', 'subject', 'setting', 'block', 'index', 'stim',
    'stim_trigger', 'Offers_Other', 'Offers_You', 'Offer_Trigger',
    'reaction', 'RT', 'emotion', 'event_id', 'epoch', 'time'
]

def extract_expressor(stim):
    if pd.isna(stim):
        return None
    match = re.search(r'(Fema\d+|Male\d+)', str(stim))
    return match.group(1) if match else None

def calc_offer_type(other, you):
    try:
        pair = f"{int(float(other))}:{int(float(you))}"
    except Exception:
        return "other"
    fair_set = ["5:5", "6:4", "4:6"]
    unfair_set = ["9:1", "1:9", "8:2", "2:8"]
    if pair in fair_set:
        return "fair"
    elif pair in unfair_set:
        return "unfair"
    else:
        return "other"

print("\n====== 单被试文件清洗及试次数统计（不合并大表！）======")
files = [f for f in os.listdir(phase_folder) if 'epo' in f and f.endswith('.csv')]
print("实际找到文件：", files)

trial_counts_all = []
other_offer_trials = []

for fname in tqdm(files, desc='单被试文件清洗与统计'):
    fpath = os.path.join(phase_folder, fname)
    df = pd.read_csv(fpath)
    participant_id = fname.split('_')[0]
    if "participant_id" not in df.columns:
        df['participant_id'] = participant_id
    # ===== 保留所有meta列 + 所有EEG电极列 =====
    all_eeg_cols = [col for col in df.columns if col not in meta_cols]
    needed_cols = meta_cols + all_eeg_cols
    needed_cols = [col for col in needed_cols if col in df.columns]
    df = df[needed_cols]

    # ===== 添加 offer_type 等新信息 =====
    df['offer_type'] = df.apply(lambda row: calc_offer_type(row['Offers_Other'], row['Offers_You']), axis=1)
    df['offer_ratio'] = df['Offers_Other'].astype(str) + ':' + df['Offers_You'].astype(str)
    df['expressor'] = df['stim'].apply(extract_expressor)

    # ===== 找出RT极端值 trial index =====
    bad_trial_indices_rt = df.loc[(df['RT'] < 150) | (df['RT'] >= 3000), 'index'].unique()

    # ===== 仅用ROI电极点做5SD极端值剔除（任何一个 ROI 通道超5SD，该 trial 全部 time 都删）=====
    roi_eeg_cols = [col for col in all_eeg_cols if col in [
        'F3', 'Fz', 'F4', 'FC1', 'FC2', 'Cz', 'Pz', 'C1', 'C2', 'CP1', 'CP2']]
    bad_trial_indices_erp = set()
    for idx in df['index'].unique():
        trial_df = df[df['index'] == idx]
        for col in roi_eeg_cols:
            vals = trial_df[col]
            mu, sd = vals.mean(), vals.std()
            if sd == 0: continue
            if any((vals < mu - 5 * sd) | (vals > mu + 5 * sd)):
                bad_trial_indices_erp.add(idx)
                break

    all_bad_trial_indices = set(bad_trial_indices_rt).union(bad_trial_indices_erp)
    df_clean = df[~df['index'].isin(all_bad_trial_indices)].copy().reset_index(drop=True)

    # ===== 检查“other”类型 =====
    others = df_clean[df_clean['offer_type'] == 'other']
    if not others.empty:
        other_offer_trials.append((participant_id, others[['Offers_Other', 'Offers_You', 'offer_ratio', 'index']]))

    # ===== 保存单被试 clean 文件（完整所有time行！） =====
    clean5sd_file = os.path.join(cleaned_folder, fname.replace('.csv', '_Clean5SD.csv'))
    df_clean.to_csv(clean5sd_file, index=False, encoding='utf-8-sig')

    # ===== 统计该被试每个条件的 trial 数 =====
    trials_count = (
        df_clean[df_clean['offer_type'].isin(['fair', 'unfair'])]
        .groupby(['participant_id', 'setting', 'emotion', 'offer_type'])['index']
        .nunique()
        .reset_index()
        .rename(columns={'index': 'n_trial'})
    )
    trial_counts_all.append(trials_count)

# ===== 合并所有被试的 trial 统计 =====
trial_counts_all_df = pd.concat(trial_counts_all, ignore_index=True)
trial_count_file = os.path.join(phase_folder, 'trial_count_by_condition.csv')
trial_counts_all_df.to_csv(trial_count_file, index=False, encoding='utf-8-sig')
print(f"\n所有被试各条件trial计数文件已输出：{trial_count_file}")

# ===== 输出other类型的试次检查（可选）=====
if other_offer_trials:
    print("\n====== 有未被识别为fair/unfair的offer类型，请核查下列内容：======")
    for pid, df_oth in other_offer_trials:
        print(f"\n被试 {pid} 中的other类型 offers：")
        print(df_oth.value_counts(['Offers_Other','Offers_You']))
else:
    print("\n所有offer类型均被正确识别为fair/unfair，没有other类型！")

print("全部数据清洗、极端值剔除、唯一性保障与trial计数统计已完成！")
