import os
import numpy as np
import pandas as pd
from scipy.io import savemat

data_path = r"E:\PD_E1_UG_jg\EEG_R_Python_Pipeline_JG_Backup\E1_UG\epochs\offer_phase\cleaned"
print("实际文件列表：", os.listdir(data_path))
output_path = r"E:\PD_E1_UG_jg\EEG_R_Python_Pipeline_JG_Backup\E1_UG\epochs\offer_phase\epochs_mat\RIDE_raw_mat"
os.makedirs(output_path, exist_ok=True)

brain_channels = [
    'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F9', 'F7', 'F5', 'F3', 'Fz', 'F4', 'F6', 'F8', 'F10',
    'FT7', 'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P7', 'P5', 'P3', 'Pz', 'P4', 'P6', 'P8',
    'PO9', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'PO10',
    'O1', 'Oz', 'O2'
]
epoch_start = -500
epoch_end = 1500
sfreq = 500
epoch_len = epoch_end - epoch_start
epoch_samples = int(epoch_len * sfreq / 1000)

files = [f for f in os.listdir(data_path) if f.lower().endswith('_clean5sd.csv')]


print(f"发现 {len(files)} 个文件：{files}")

for file in files:
    try:
        print(f"\n正在处理: {file}")
        df = pd.read_csv(os.path.join(data_path, file))
        used_channels = [ch for ch in brain_channels if ch in df.columns]
        print(f"可用脑电通道: {used_channels}")
        n_chan = len(used_channels)
        n_timepoints = epoch_samples
        total_points = df.shape[0]
        n_trials = total_points // n_timepoints
        used_points = n_trials * n_timepoints

        if n_chan == 0:
            print("没有可用EEG通道，跳过！")
            continue

        if used_points == 0 or n_trials == 0:
            print("数据不足一个完整trial，跳过！")
            continue

        if used_points != total_points:
            print(f"Warning: 有 {total_points-used_points} 行被丢弃")
            df = df.iloc[:used_points, :]

        X = df[used_channels].values.reshape(n_trials, n_timepoints, n_chan)
        # ==== 基线矫正区间（单位ms） ====
        baseline_start = -200
        baseline_end = 0

        # 计算基线区间对应采样点索引
        baseline_idx = [
            i for i in range(n_timepoints)
            if (epoch_start + i * 1000 / sfreq) >= baseline_start and
            (epoch_start + i * 1000 / sfreq) <= baseline_end
        ]

        # === 对每个 trial 做基线校正 ===
        for t in range(n_trials):
            baseline = X[t, baseline_idx, :].mean(axis=0)  # [n_chan]
            X[t, :, :] = X[t, :, :] - baseline  # 所有时间点都减去自己的基线
            
        EEG_data = np.transpose(X, (1, 2, 0))

        trial_meta = df.iloc[::n_timepoints, :].reset_index(drop=True)

        # 直接读取表里的offer_type/offer_ratio
        offer_type = trial_meta['offer_type'].to_numpy().reshape(-1,1)
        offer_ratio = trial_meta['offer_ratio'].astype(str).to_numpy().reshape(-1,1)

        mat_data = {
            'EEG_data': EEG_data,
            'chan_names': np.array(used_channels),
            'RTs': trial_meta['RT'].to_numpy().reshape(-1,1),
            'emotion': trial_meta['emotion'].astype(str).to_numpy().reshape(-1,1),
            'offer_type': offer_type,
            'offer_ratio': offer_ratio,
            'index': trial_meta['index'].to_numpy().reshape(-1,1),
            'setting': trial_meta['setting'].to_numpy().reshape(-1,1),
            'reaction': trial_meta['reaction'].to_numpy().reshape(-1,1),
            'block': trial_meta['block'].to_numpy().reshape(-1,1),
            'subject': trial_meta['subject'].to_numpy().reshape(-1,1),
            'participant_id': trial_meta['participant_id'].to_numpy().reshape(-1,1),
        }
        outname = file.replace('.csv', '.mat')
        savemat(os.path.join(output_path, outname), mat_data)
        print(f'已保存到 {os.path.join(output_path, outname)}')
    except Exception as e:
        print(f"处理 {file} 时出错：{e}")

print("\n全部完成！")
