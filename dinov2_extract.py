import os
# ffmpeg多线程解码，解决cv2读取视频慢！！！
os.environ["OPENCV_FFMPEG_THREADS"] = "8"
# 解决cuda内存碎片OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import cv2
import torch
from torchvision import transforms


DATA_ROOT = "/data/Wcy/code/vad_indoor/UCF_indoor"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# ==========新增开启TF32加速==========
if device == "cuda":
    torch.set_float32_matmul_precision('high')

print("加载 DINOv2 模型...")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
model.eval()
# if device == "cuda":
#     model = torch.compile(model, mode="reduce-overhead")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def extract_features_from_video(video_path, target_fps=4):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(fps / target_fps)))

    batch_size = 28
    cpu_frames = []
    idx = 0
    features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cpu_frames.append(transform(frame_rgb))
            # 攒够一批送入GPU推理
            if len(cpu_frames) >= batch_size:
                batch_tensor = torch.stack(cpu_frames).to(device)
                with torch.no_grad():
                    feat_batch = model(batch_tensor)
                    feat_batch = torch.nn.functional.normalize(feat_batch, p=2, dim=-1)
                features.append(feat_batch.cpu())
                del batch_tensor, feat_batch
                torch.cuda.empty_cache()
                cpu_frames.clear()
        idx += 1
    cap.release()

    # 处理剩余不足一个batch的帧
    if len(cpu_frames) > 0:
        batch_tensor = torch.stack(cpu_frames).to(device)
        with torch.no_grad():
            feat_batch = model(batch_tensor)
            feat_batch = torch.nn.functional.normalize(feat_batch, p=2, dim=-1)
        features.append(feat_batch.cpu())
        del batch_tensor, feat_batch
        torch.cuda.empty_cache()
        cpu_frames.clear()

    if len(features) == 0:
        return None
    return torch.cat(features, dim=0)


if __name__ == "__main__":
    save_root = os.path.join(DATA_ROOT, "features")
    os.makedirs(os.path.join(save_root, "normal"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "abnormal"), exist_ok=True)

    # ========== 真实UCF‑Crime原始视频路径 ==========
    ucf_dataset_root = "/data/Wcy/UCF/OpenDataLab___UCF-Crime/raw/UCF-Crime/Anomaly-Detection-Dataset"

    # normal视频：两个文件夹
    normal_dirs = [
        os.path.join(ucf_dataset_root, "Training-Normal-Videos-Part-1"),
        os.path.join(ucf_dataset_root, "Training-Normal-Videos-Part-2"),
    ]
    normal_video_list = []
    for d in normal_dirs:
        normal_video_list.extend(glob.glob(os.path.join(d, "*.mp4")))

    # abnormal视频：Anomaly‑Videos‑Part1~4下各个子文件夹
    abnormal_video_list = []
    for part_name in ["Anomaly-Videos-Part-1","Anomaly-Videos-Part-2","Anomaly-Videos-Part-3","Anomaly-Videos-Part-4"]:
        part_path = os.path.join(ucf_dataset_root, part_name)
        if not os.path.exists(part_path):
            continue
        subfolders = [f for f in os.listdir(part_path) if os.path.isdir(os.path.join(part_path,f))]
        for sub in subfolders:
            sub_path = os.path.join(part_path, sub)
            abnormal_video_list.extend(glob.glob(os.path.join(sub_path,"*.mp4")))

    print(f"\n--- 开始处理类别: normal (共 {len(normal_video_list)} 个视频) ---")
    for v_path in normal_video_list:
        v_name = os.path.splitext(os.path.basename(v_path))[0]
        out_path = os.path.join(save_root, "normal", f"{v_name}.pt")
        if os.path.exists(out_path):
            print(f"[跳过已存在] {v_name}")
            continue
        print(f"正在提取: {v_name} ...", end="", flush=True)
        feat = extract_features_from_video(v_path)
        if feat is not None:
            torch.save(feat, out_path)
            print(f" 完成! 形状: {feat.shape}")
        else:
            print(" 失败: 未读取到视频帧")

    print(f"\n--- 开始处理类别: abnormal (共 {len(abnormal_video_list)} 个视频) ---")
    for v_path in abnormal_video_list:
        v_name = os.path.splitext(os.path.basename(v_path))[0]
        out_path = os.path.join(save_root, "abnormal", f"{v_name}.pt")
        if os.path.exists(out_path):
            print(f"[跳过已存在] {v_name}")
            continue
        print(f"正在提取: {v_name} ...", end="", flush=True)
        feat = extract_features_from_video(v_path)
        if feat is not None:
            torch.save(feat, out_path)
            print(f" 完成! 形状: {feat.shape}")
        else:
            print(" 失败: 未读取到视频帧")

    print("\n==== 所有视频特征提取完毕！====")
