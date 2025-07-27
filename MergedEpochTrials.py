import os
import pandas as pd
import re
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import numpy as np

# ========== 配置参数 ==========

root_dir = r"C:\EEG_R_Python_Pipeline_JG\E1_UG"
phases = ["offer_phase", "face_phase"]

# ROI电极配置
roi_dict = {
    'FRN':       ['F3', 'Fz', 'F4', 'FC1', 'FC2', 'Cz'],
    'LPP_offer': ['Pz', 'Cz', 'C1', 'C2', 'CP1', 'CP2'],
    'P1':        ['O1', 'O2', 'Oz', 'PO7', 'PO8'],
    'N170':      ['TP9', 'TP10', 'P7', 'P8', 'PO9', 'PO10', 'O1', 'O2'],
    'EPN':       ['PO7', 'PO8', 'PO9', 'PO10', 'TP9', 'TP10'],
    'LPP_face':  ['Pz', 'Cz', 'C1', 'C2', 'CP1', 'CP2'],
}

# 自动汇总所有ERP成分的ROI电极点
roi_set = set()
for v in roi_dict.values():
    roi_set.update(v)
roi_electrodes = sorted(list(roi_set))

# meta信息和时间
meta_cols = [
    'participant_id', 'subject', 'setting', 'block', 'index', 'stim',
    'stim_trigger', 'Offers_Other', 'Offers_You', 'Offer_Trigger',
    'reaction', 'RT', 'emotion', 'event_id', 'epoch', 'time'
]

covariate_path = os.path.join(root_dir, "data", "SVO_PID5BF_PostRating.xlsx")
covariate_cols = [
    "SVO_angle", "Negative_Affectivity", "Detachment", "Antagonism",
    "Disinhibition", "Anankastia", "Psychoticism", "Rating_1", "Rating_2", "Rating_3"
]

erp_windows = {
    "offer_phase": {
        'FRN':        (0.250, 0.300),      
        'LPP_offer':  (0.400, 0.600),
    },
    "face_phase": {
        'P1':        (0.080, 0.130),
        'N170':      (0.130, 0.200),
        'EPN':       (0.200, 0.350),
        'LPP_face':  (0.400, 0.600),
    }
}

trial_key_cols = [
    'participant_id', 'subject', 'setting', 'block', 'index', 'stim',
    'stim_trigger', 'Offers_Other', 'Offers_You', 'Offer_Trigger',
    'reaction', 'RT', 'emotion', 'event_id', 'epoch'
]

# ========== 工具函数 ==========

def extract_expressor(stim):
    if pd.isna(stim):
        return None
    match = re.search(r'(Fema\d+|Male\d+)', str(stim))
    return match.group(1) if match else None

def calc_offer_type(row):
    other, you = row.get('Offers_Other'), row.get('Offers_You')
    fair_set = [(5, 5), (6, 4)]
    unfair_set = [(8, 2), (9, 1)]
    if (other, you) in fair_set:
        return "fair"
    elif (other, you) in unfair_set:
        return "unfair"
    return None

def merge_covariates(trials, covariate_path, covariate_cols):
    subject_vars = pd.read_excel(covariate_path)
    subject_vars_keep = subject_vars[["participant_id"] + covariate_cols].copy()
    scaler = StandardScaler()
    for col in covariate_cols:
        subject_vars_keep[col + "_z"] = scaler.fit_transform(subject_vars_keep[[col]].astype(float))
    subject_vars_z = subject_vars_keep[["participant_id"] + [col + "_z" for col in covariate_cols]]
    trials_merged = pd.merge(trials, subject_vars_z, how="left", on="participant_id")
    return trials_merged

def calc_window_means_with_roi(df, erp_win_dict, roi_dict, trial_keys):
    means_df = None
    for erp, (tmin, tmax) in erp_win_dict.items():
        roi_chans = roi_dict.get(erp, [])
        missing_roi = [chan for chan in roi_chans if chan not in df.columns]
        if missing_roi:
            print(f"ERP: {erp} 缺失ROI通道: {missing_roi}，跳过该成分！")
            continue
        sub = df[(df['time'] >= tmin) & (df['time'] <= tmax)].copy()
        if sub.empty:
            print(f"ERP: {erp} 时间窗数据为空，跳过。")
            continue
        # 对ROI通道按行均值
        sub[f'{erp}_ROImean'] = sub[roi_chans].mean(axis=1)
        use_keys = [col for col in trial_keys if col in sub.columns]
        erp_mean = (
            sub.groupby(use_keys)[f'{erp}_ROImean']
            .mean()
            .reset_index()
            .rename(columns={f'{erp}_ROImean': f'{erp}'})
        )
        if means_df is None:
            means_df = erp_mean
        else:
            means_df = pd.merge(means_df, erp_mean, on=use_keys, how='outer')
    if means_df is not None:
        means_df = means_df.drop_duplicates()
    return means_df

def remove_outliers_5sd(df, erp_cols):
    drop_idx = set()
    for col in erp_cols:
        mu, sd = df[col].mean(), df[col].std()
        idx = df[(df[col] < mu - 5 * sd) | (df[col] > mu + 5 * sd)].index
        print(f"{col}: 剔除极端值trial数={len(idx)}")
        drop_idx.update(idx)
    df_clean = df.drop(list(drop_idx)).reset_index(drop=True)
    return df_clean, drop_idx

def summarize_trial_counts(window_means, clean_means, group_vars, phase_folder):
    count_rt = window_means.groupby(group_vars).size().reset_index(name='n_trial_after_RT')
    count_5sd = clean_means.groupby(group_vars).size().reset_index(name='n_trial_after_5SD')
    summary = pd.merge(count_rt, count_5sd, on=group_vars, how='outer').fillna(0)
    summary['n_trial_after_RT'] = summary['n_trial_after_RT'].astype(int)
    summary['n_trial_after_5SD'] = summary['n_trial_after_5SD'].astype(int)
    summary_file = os.path.join(phase_folder, 'trial_count_per_condition.csv')
    summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"【汇总表已导出: {summary_file}】")
    return summary

# ========== 主流程 ==========

for phase in phases:
    print(f"\n====== 正在处理 phase: {phase} ======")
    phase_folder = os.path.join(root_dir, "epochs", phase)
    cleaned_folder = os.path.join(phase_folder, "cleaned")
    if phase == "offer_phase":
        os.makedirs(cleaned_folder, exist_ok=True)
    outpath = os.path.join(phase_folder, "epoch_trials.csv")
    files = [f for f in os.listdir(phase_folder) if f.endswith('.csv') and f.startswith('Vp')]
    dfs, removed_trials_list = [], []

    # ========== 1. 合并clean文件，仅保留meta和ROI电极点 ==========
    for fname in tqdm(files, desc=f'Processing {phase}'):
        fpath = os.path.join(phase_folder, fname)
        df = pd.read_csv(fpath)
        participant_id = fname.split('_')[0]
        if "participant_id" not in df.columns:
            df['participant_id'] = participant_id
        # 只保留meta+ROI通道
        needed_cols = [col for col in (meta_cols + roi_electrodes) if col in df.columns]
        df = df[needed_cols]
        # 计算meta衍生列
        if phase == "offer_phase":
            df['offer_type'] = df.apply(calc_offer_type, axis=1)
            df['offer_ratio'] = df['Offers_Other'].astype(str) + ':' + df['Offers_You'].astype(str)
        df['expressor'] = df['stim'].apply(extract_expressor)
        dfs.append(df)
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(outpath, index=False, encoding='utf-8-sig')
    print(f"{phase} 合并后的clean数据已保存为：{outpath}\n")

    # ========== 剔除RT极端trial（仅offer_phase需） ==========
    if phase == "offer_phase":
        bad_trial_indices = merged_df.loc[(merged_df['RT'] < 150) | (merged_df['RT'] >= 3000), 'index'].unique()
        before = merged_df['index'].nunique()
        merged_df = merged_df[~merged_df['index'].isin(bad_trial_indices)].copy()
        after = merged_df['index'].nunique()
        print(f"offer_phase: 剔除极端RT trial数：{before - after}")

    # ========== 2. 窗口+ROI均值 ==========
    print("计算窗口+ROI均值并剔除极端值...")
    if phase == "offer_phase":
        keys_to_use = [col for col in (trial_key_cols + ['offer_type', 'offer_ratio', 'expressor']) if col in merged_df.columns]
    else:
        keys_to_use = [col for col in (trial_key_cols + ['expressor']) if col in merged_df.columns]
    window_means = calc_window_means_with_roi(merged_df, erp_windows[phase], roi_dict, keys_to_use)
    mean_outfile = os.path.join(phase_folder, "epoch_trial_window_means_ROI.csv")
    if window_means is None or window_means.empty:
        raise ValueError("未计算出任何ERP窗口均值（请确认数据内包含正确的ROI通道和采样点）！")
    window_means.to_csv(mean_outfile, index=False, encoding='utf-8-sig')
    print(f"窗口ROI均值文件已输出：{mean_outfile}")

    erp_cols = [col for col in window_means.columns if col in erp_windows[phase].keys()]
    clean_means, drop_idx = remove_outliers_5sd(window_means, erp_cols)
    clean_outfile = os.path.join(phase_folder, "epoch_trial_window_means_ROI_clean5SD.csv")
    clean_means.to_csv(clean_outfile, index=False, encoding='utf-8-sig')
    print(f"窗口均值5SD剔除后文件已输出：{clean_outfile}")
    if drop_idx:
        window_means.loc[list(drop_idx)].to_csv(os.path.join(phase_folder, "removed_trial_means_5SD.csv"), index=False, encoding='utf-8-sig')

    # ========== 3. 合并协变量（仅offer_phase需，如face_phase要可取消注释） ==========
    if phase == "offer_phase":
        print("正在合并协变量并标准化到最终trial表...")
        clean_means_with_cov = merge_covariates(clean_means, covariate_path, covariate_cols)
        outpath_cov = os.path.join(phase_folder, "epoch_trial_window_means_ROI_clean5SD_with_covariates.csv")
        clean_means_with_cov.to_csv(outpath_cov, index=False, encoding='utf-8-sig')
        print(f"最终trial均值表（含协变量）已保存为：{outpath_cov}")

    # ========== 4. trial计数表 ==========
    print("正在统计每被试每条件trial数...")
    if phase == "offer_phase":
        group_vars = [col for col in ['participant_id', 'setting', 'emotion', 'offer_type'] if col in window_means.columns]
    else:
        group_vars = [col for col in ['participant_id', 'setting', 'emotion'] if col in window_means.columns]
    summarize_trial_counts(window_means, clean_means, group_vars, phase_folder)
    print(f"{phase} 数据处理完毕。\n")

print("【所有phase数据合并、ROI均值与极端值处理与trial计数全部完成！】")
