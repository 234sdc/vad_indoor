import os
import glob
import json
import cv2
import torch
import torch.nn as nn
import numpy as np

DATA_ROOT = "/data/Wcy/code/vad_indoor/UCF_indoor"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
manifest_path = os.path.join(DATA_ROOT,"candidate_clips_manifest.json")

# ====================新增：manifest存在直接跳过切片====================
if os.path.exists(manifest_path):
    print(f"✅ {manifest_path} 已存在，跳过TCN推理和切片生成！")
    exit(0)
# ======================================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding !=0 else out

class TemporalConvNet(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=128, num_layers=4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            c_in = in_dim if i ==0 else hidden_dim
            layers.append(CausalConv1d(c_in, hidden_dim, kernel_size=3, dilation=dilation))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Conv1d(hidden_dim,1,kernel_size=1), nn.Sigmoid())
    def forward(self, x):
        feat = self.network(x)
        score = self.head(feat)
        return score.squeeze(1)

def gaussian_smooth(scores, sigma=1.5, window_size=5):
    radius = window_size //2
    x = np.arange(-radius, radius+1)
    kernel = np.exp(-0.5 * (x/sigma)**2)
    kernel /= kernel.sum()
    padded = np.pad(scores, (radius, radius), mode='edge')
    return np.convolve(padded, kernel, mode='valid')

# 调低阈值，方便先跑出候选片段
def detect_intervals(scores, tau_high=0.0004, tau_low=0.0002):
    intervals = []
    in_anomaly = False
    start_idx = 0
    for t, s in enumerate(scores):
        if not in_anomaly and s >= tau_high:
            in_anomaly = True
            start_idx = t
        elif in_anomaly and s < tau_low:
            in_anomaly = False
            intervals.append((start_idx, t))
    if in_anomaly:
        intervals.append((start_idx, len(scores)-1))
    return intervals

def crop_video(video_path, out_path, start_sec, end_sec):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cur = start_frame
    while cap.isOpened() and cur <= end_frame:
        ret,frm = cap.read()
        if not ret:break
        out.write(frm)
        cur += 1
    cap.release()
    out.release()

def find_ucf_abnormal_video(vid_name):
    abnormal_root = "/data/Wcy/UCF/OpenDataLab___UCF-Crime/raw/UCF-Crime/Anomaly-Detection-Dataset"
    for part in ["Anomaly-Videos-Part-1","Anomaly-Videos-Part-2","Anomaly-Videos-Part-3","Anomaly-Videos-Part-4"]:
        part_path = os.path.join(abnormal_root, part)
        if not os.path.exists(part_path):
            continue
        for subdir in os.listdir(part_path):
            sub_path = os.path.join(part_path, subdir)
            if not os.path.isdir(sub_path):
                continue
            candidate = os.path.join(sub_path, f"{vid_name}.mp4")
            if os.path.exists(candidate):
                return candidate
    return None

if __name__ == "__main__":
    # 修改：读取tcn_best.pth，和train_tcn输出一致
    model_path = os.path.join(DATA_ROOT,"models","tcn_best.pth")
    model = TemporalConvNet().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    clip_out_dir = os.path.join(DATA_ROOT,"candidate_clips")
    os.makedirs(clip_out_dir, exist_ok=True)
    feat_abnormal = glob.glob(os.path.join(DATA_ROOT,"features/abnormal/*.pt"))
    sample_fps = 4.0
    buffer_sec = 1.0
    clip_records = []
    print(f"\n--- 粗筛处理 {len(feat_abnormal)} 异常视频特征 ---")

    with torch.no_grad():
        for fpath in feat_abnormal:
            v_name = os.path.splitext(os.path.basename(fpath))[0]
            raw_video = find_ucf_abnormal_video(v_name)
            if raw_video is None:
                print(f"警告：找不到原始视频 {v_name}.mp4，跳过")
                continue
            feat = torch.load(fpath, weights_only=True).to(device)
            feat = feat.unsqueeze(0).transpose(1,2)

            # =========核心修复=========
            raw_scores = model(feat).squeeze(0).cpu().numpy()
            # 删掉原来 raw_scores = 1.0 - raw_normal_prob
            # 模型输出直接就是异常分数，越大越异常
            # ==========================

            smoothed_scores = gaussian_smooth(raw_scores)
            intervals = detect_intervals(smoothed_scores)
            print(f"\n视频[{v_name}] 最高分 raw:{raw_scores.max():.4f} smooth:{smoothed_scores.max():.4f} 命中区间数:{len(intervals)}")

            for idx,(s_idx,e_idx) in enumerate(intervals):
                peak_sub_idx = s_idx + np.argmax(smoothed_scores[s_idx:e_idx+1])
                peak_score = float(smoothed_scores[peak_sub_idx])
                s_sec = max(0.0, (s_idx/sample_fps)-buffer_sec)
                e_sec = (e_idx/sample_fps)+buffer_sec
                if (e_sec - s_sec) >8.0:
                    peak_sec = peak_sub_idx / sample_fps
                    s_sec = max(0.0, peak_sec-3.0)
                    e_sec = peak_sec + 3.0
                clip_len = e_sec - s_sec
                clip_name = f"{v_name}_clip_{idx:02d}.mp4"
                clip_save = os.path.join(clip_out_dir, clip_name)
                crop_video(raw_video, clip_save, s_sec, e_sec)
                rec = {
                    "clip_name":clip_name,
                    "source_video":f"{v_name}.mp4",
                    "start_time_sec":round(s_sec,2),
                    "end_time_sec":round(e_sec,2),
                    "duration_sec":round(clip_len,2),
                    "S_coarse":round(peak_score,4)
                }
                clip_records.append(rec)
                print(f"  -> {clip_name}  {s_sec:.1f}~{e_sec:.1f}s S_coarse={peak_score:.3f}")

    with open(manifest_path,"w",encoding="utf-8") as f:
        json.dump(clip_records,f,indent=2,ensure_ascii=False)
    print(f"\n切片完成，共 {len(clip_records)} 候选切片，清单：{manifest_path}")
