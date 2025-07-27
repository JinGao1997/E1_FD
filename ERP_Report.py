# 🟢 标签整理
import os
import pandas as pd

root_dir = r"E:\PD_E1_UG_jg\EEG_R_Python_Pipeline_JG_Backup\E1_UG\offerphase400600"
phases = ["offer_phase", "face_phase"]

def add_labels_offer(input_path, output_path):
    df = pd.read_csv(input_path)
    df['emotion'] = df['label'].str.split("_").str[1]
    df['offer_type'] = df['label'].apply(lambda x: (
        "fair" if x.split("_")[2:4] in [["5", "5"], ["6", "4"]] else
        "unfair" if x.split("_")[2:4] in [["8", "2"], ["9", "1"]] else None
    ))
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Processed (offer_phase): {output_path}")

def add_labels_face(input_path, output_path):
    df = pd.read_csv(input_path)
    df['emotion'] = df['label'].str.split("_").str[1]
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Processed (face_phase): {output_path}")

if __name__ == "__main__":
    for phase in phases:
        phase_dir = os.path.join(root_dir, phase)
        for fname, outname in [("ave.csv", "ave_with_labels.csv"), ("grand_ave.csv", "grand_ave_GROUPLEVEL.csv")]:
            in_path = os.path.join(phase_dir, fname)
            out_path = os.path.join(phase_dir, outname)
            if os.path.exists(in_path):
                if phase == "offer_phase":
                    add_labels_offer(in_path, out_path)
                else:
                    add_labels_face(in_path, out_path)
            else:
                print(f"File not found, skipped: {in_path}")




# 波形图
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== 公共参数 ==========
root_dir = r"E:\PD_E1_UG_jg\EEG_R_Python_Pipeline_JG_Backup\E1_UG\offerphase400600"
full_label = {'dis': 'Disgust', 'dom': 'Dominance', 'neu': 'Neutral', 'aff': 'Affiliative', 'enj': 'Reward'}
emotion_colors = {'dis': "#755627", 'dom': "#F5900C", 'neu': "#C5C5C5EC", 'aff': "#39E04F", 'enj': "#FC0000"}
emotions = list(full_label.keys())
offer_types = ['fair', 'unfair']

# # ========== FACE PHASE ==========

# face_phase = "face_phase"
# face_data_file = os.path.join(root_dir, face_phase, "ave_with_labels.csv")
# face_save_dir = os.path.join(root_dir, face_phase, "figures_waveforms")
# os.makedirs(face_save_dir, exist_ok=True)

# face_erp_settings = {
#     "P1": {"roi": ["O1", "O2", "Oz", "PO7", "PO8"], "win": (0.08, 0.13), "ylabel": "P1 (µV)"},
#     "N170": {"roi": ["TP9", "TP10", "P7", "P8", "PO9", "PO10", "O1", "O2"], "win": (0.13, 0.20), "ylabel": "N170 (µV)"},
#     "EPN": {"roi": ["PO7", "PO8", "PO9", "PO10", "TP9", "TP10"], "win": (0.20, 0.35), "ylabel": "EPN (µV)"},
#     "LPP_face": {"roi": ["Pz", "Cz", "C1", "C2", "CP1", "CP2"], "win": (0.4, 0.60), "ylabel": "LPP (µV)"},
# }

# def plot_erp_by_emotion_auto(df, erp, roi_chans, win, ylab, title, ylim=None, save_dir=None):
#     df['roi_mean'] = df[roi_chans].mean(axis=1)
#     mean_df = df.groupby(['emotion', 'time'])['roi_mean'].mean().reset_index()
#     fig, ax = plt.subplots(figsize=(7, 4.2), facecolor='white')
#     for emo in emotions:
#         d = mean_df[mean_df['emotion'] == emo]
#         if not d.empty:
#             ax.plot(
#                 d['time']*1000, d['roi_mean'],
#                 color=emotion_colors[emo], lw=1.5, linestyle='-',
#                 label=full_label[emo]
#             )
#     ax.axvspan(win[0]*1000, win[1]*1000, color='grey', alpha=0.06, lw=0)
#     ax.axvline(0, ls='--', color='grey', lw=1)
#     ax.axhline(0, ls=':', color='grey', lw=1)
#     ax.set_xlim([-200, 1000])
#     if ylim is not None:
#         ax.set_ylim(ylim)
#     ax.set_xlabel('Time (ms)', fontsize=7)
#     ax.set_ylabel(ylab, fontsize=7)
#     ax.set_title(title, fontsize=10, weight='bold', pad=4)
#     ax.set_xticks(np.arange(-200, 1100, 100))
#     ax.tick_params(axis='x', labelsize=9)
#     ax.tick_params(axis='y', labelsize=9)
#     leg = ax.legend(
#         title="Emotion", fontsize=8, title_fontsize=9,
#         frameon=True, fancybox=True, ncol=1, framealpha=0.85,
#         loc='center left', bbox_to_anchor=(1.00, 0.52), borderaxespad=0.5, handlelength=1.5
#     )
#     leg.get_frame().set_edgecolor('lightgrey')
#     plt.tight_layout(rect=[0, 0, 0.98, 1])

#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#         fname = os.path.join(save_dir, f"{erp}_ROI_waveform.tif")
#         plt.savefig(fname, dpi=400, bbox_inches='tight', format='tiff')
#     plt.close(fig)

# df_face = pd.read_csv(face_data_file)
# for erp, info in face_erp_settings.items():
#     roi = info['roi']
#     win = info['win']
#     ylab = info['ylabel']
#     temp = df_face.groupby(['emotion', 'time'])[roi].mean().mean(axis=1)
#     ylim = (temp.min() - 0.3, temp.max() + 0.3)
#     title = f"{erp} ROI Waveforms"
#     plot_erp_by_emotion_auto(df_face, erp, roi, win, ylab, title, ylim=ylim, save_dir=face_save_dir)
# print(f"全部 {face_phase} ROI波形图自动保存到: {face_save_dir}")


# ========== OFFER PHASE ==========

offer_phase = "offer_phase"
offer_data_file = os.path.join(root_dir, offer_phase, "ave_with_labels.csv")
offer_save_dir = os.path.join(root_dir, offer_phase, "figures_waveforms")
os.makedirs(offer_save_dir, exist_ok=True)

offer_erp_settings = {
    "FRN": {"roi": ["F3", "Fz", "F4", "FC1", "FC2", "Cz"], "win": (0.25, 0.30), "ylabel": "FRN (µV)"},
    "LPP_offer": {"roi": ["Pz", "Cz", "C1", "C2", "CP1", "CP2"], "win": (0.40, 0.60), "ylabel": "LPP (µV)"}
}

def plot_fair_vs_unfair_by_emotion(df, erp, ylab, win, ylim=None, save_dir=None):
    for emo in emotions:
        fig, ax = plt.subplots(figsize=(7, 4.2), facecolor='white')
        for offer_type in offer_types:
            d = df[(df['offer_type'] == offer_type) & (df['emotion'] == emo) & (df['time']*1000 >= -200) & (df['time']*1000 <= 1000)]
            if not d.empty:
                linestyle = '-.' if offer_type == 'fair' else '-'
                lw = 1.5
                label = 'Fair' if offer_type == 'fair' else 'Unfair'
                ax.plot(
                    d['time']*1000, d['mean'],
                    color=emotion_colors[emo], lw=lw, linestyle=linestyle, label=label
                )
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
            os.makedirs(save_dir, exist_ok=True)
            fname = os.path.join(save_dir, f"{erp}_{emo}_FairVsUnfair_Waveform.tif")
            plt.savefig(fname, dpi=400, bbox_inches='tight', format='tiff')
        plt.close(fig)

def plot_all_emotion_by_condition(df, erp, ylab, win, offer_type, ylim=None, save_dir=None):
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor='white')
    linestyle = '-.' if offer_type == 'fair' else '-'
    lw = 1.5
    for emo in emotions:
        d = df[(df['offer_type'] == offer_type) & (df['emotion'] == emo) & (df['time']*1000 >= -200) & (df['time']*1000 <= 1000)]
        if not d.empty:
            ax.plot(
                d['time']*1000, d['mean'],
                color=emotion_colors[emo], lw=lw, linestyle=linestyle,
                label=full_label[emo]
            )
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
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f"{erp}_{offer_type}_AllEmo_Waveform.tif")
        plt.savefig(fname, dpi=400, bbox_inches='tight', format='tiff')
    plt.close(fig)

def plot_fair_unfair_across_emotions(df, erp, ylab, win, ylim=None, save_dir=None):
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor='white')
    for offer_type in offer_types:
        d = df[df['offer_type'] == offer_type]
        mean_over_emo = d.groupby('time')['mean'].mean().reset_index()
        style = '-.' if offer_type == 'fair' else '-'
        lw = 1.5
        label = 'Fair' if offer_type == 'fair' else 'Unfair'
        ax.plot(
            mean_over_emo['time']*1000, mean_over_emo['mean'],
            color='black', lw=lw, linestyle=style, label=label
        )
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
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f"{erp}_FairVsUnfair_AllMean_Waveform.tif")
        plt.savefig(fname, dpi=400, bbox_inches='tight', format='tiff')
    plt.close(fig)


df_offer = pd.read_csv(offer_data_file)
for erp, info in offer_erp_settings.items():
    df_offer['roi_mean'] = df_offer[info['roi']].mean(axis=1)
    mean_df = df_offer.groupby(['offer_type', 'emotion', 'time'])['roi_mean'].mean().reset_index().rename(columns={'roi_mean': 'mean'})
    sel = (mean_df['time']*1000 >= -200) & (mean_df['time']*1000 <= 1000)
    ylim = (mean_df.loc[sel, 'mean'].min() - 0.3, mean_df.loc[sel, 'mean'].max() + 0.3)
    plot_fair_vs_unfair_by_emotion(mean_df, erp, info['ylabel'], info['win'], ylim=ylim, save_dir=offer_save_dir)
    for offer_type in offer_types:
        plot_all_emotion_by_condition(mean_df, erp, info['ylabel'], info['win'], offer_type, ylim=ylim, save_dir=offer_save_dir)
    plot_fair_unfair_across_emotions(mean_df, erp, info['ylabel'], info['win'], ylim=ylim, save_dir=offer_save_dir)
print(f"[{offer_phase}] FRN和LPP_offer ROI波形图（100ms刻度）批量输出完成，见 {offer_save_dir}")
