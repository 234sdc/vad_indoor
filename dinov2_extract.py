import os

# 在导入 cv2 / torch 前设置运行环境
os.environ.setdefault("OPENCV_FFMPEG_THREADS", "8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import glob
import math
import cv2
import torch
from torchvision import transforms


# 可通过环境变量覆盖路径，默认值保持与原项目一致
DATA_ROOT = os.environ.get(
    "VAD_DATA_ROOT", "/data/Wcy/code/vad_indoor/UCF_indoor"
)
UCF_DATASET_ROOT = os.environ.get(
    "UCF_CRIME_ROOT",
    "/data/Wcy/UCF/OpenDataLab___UCF-Crime/raw/UCF-Crime/Anomaly-Detection-Dataset",
)
TARGET_FPS = 4
BATCH_SIZE = 28
FEATURE_DIM = 768

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

if device == "cuda":
    torch.set_float32_matmul_precision("high")

print("加载 DINOv2 模型...")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _infer_batch(frames):
    """将一批 CPU 图像转换为 (N, 768) 的 CPU 特征。"""
    batch_tensor = torch.stack(frames, dim=0).to(device)
    with torch.inference_mode():
        feat_batch = model(batch_tensor)
        feat_batch = torch.nn.functional.normalize(feat_batch.float(), p=2, dim=-1)

    # 明确保存为 float32，保持与后续 TCN 训练的兼容性
    result = feat_batch.cpu()
    del batch_tensor, feat_batch
    return result


def extract_features_from_video(video_path, target_fps=TARGET_FPS, batch_size=BATCH_SIZE):
    """
    按目标帧率抽帧并提取 DINOv2 特征。
    返回形状为 (T, 768) 的 CPU Tensor；视频无法读取时返回 None。
    """
    if target_fps <= 0:
        raise ValueError(f"target_fps 必须大于 0，当前值为 {target_fps}")
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须大于 0，当前值为 {batch_size}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        print(f"\n[读取失败] 无法打开视频: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        print(f"\n[警告] 视频 FPS 无效 ({fps})，按 25 FPS 处理: {video_path}")
        fps = 25.0

    # 保留原脚本的按帧间隔抽样方式，尽量避免改变已有流程的时间尺度
    interval = max(1, int(round(fps / target_fps)))
    cpu_frames = []
    feature_batches = []
    frame_idx = 0
    sampled_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cpu_frames.append(transform(frame_rgb))
                    sampled_frames += 1
                except Exception as exc:
                    print(f"\n[警告] 第 {frame_idx} 帧预处理失败，跳过该帧: {exc}")

                if len(cpu_frames) >= batch_size:
                    feature_batches.append(_infer_batch(cpu_frames))
                    cpu_frames.clear()

            frame_idx += 1
    except Exception as exc:
        print(f"\n[读取异常] {video_path}: {exc}")
        return None
    finally:
        cap.release()

    if cpu_frames:
        feature_batches.append(_infer_batch(cpu_frames))
        cpu_frames.clear()

    if not feature_batches:
        print(f"\n[提取失败] 没有成功读取到可用帧: {video_path}")
        return None

    features = torch.cat(feature_batches, dim=0).contiguous()

    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise RuntimeError(
            f"DINOv2 特征形状异常: {tuple(features.shape)}，"
            f"预期为 (T, {FEATURE_DIM})；视频: {video_path}"
        )

    if not torch.isfinite(features).all():
        raise RuntimeError(f"特征中出现 NaN/Inf: {video_path}")

    print(f" 抽取帧数: {sampled_frames}", end="")
    return features


def collect_video_paths():
    """收集 normal / abnormal 视频，并对路径排序，保证处理顺序稳定。"""
    normal_dirs = [
        os.path.join(UCF_DATASET_ROOT, "Training-Normal-Videos-Part-1"),
        os.path.join(UCF_DATASET_ROOT, "Training-Normal-Videos-Part-2"),
    ]

    normal_videos = []
    for directory in normal_dirs:
        if not os.path.isdir(directory):
            print(f"[警告] 正常视频目录不存在: {directory}")
            continue
        normal_videos.extend(glob.glob(os.path.join(directory, "*.mp4")))

    abnormal_videos = []
    for part_name in [
        "Anomaly-Videos-Part-1",
        "Anomaly-Videos-Part-2",
        "Anomaly-Videos-Part-3",
        "Anomaly-Videos-Part-4",
    ]:
        part_path = os.path.join(UCF_DATASET_ROOT, part_name)
        if not os.path.isdir(part_path):
            print(f"[警告] 异常视频目录不存在，跳过: {part_path}")
            continue

        for sub_path in sorted(glob.glob(os.path.join(part_path, "*"))):
            if os.path.isdir(sub_path):
                abnormal_videos.extend(glob.glob(os.path.join(sub_path, "*.mp4")))

    return sorted(set(normal_videos)), sorted(set(abnormal_videos))


def process_video_list(video_list, category, output_dir):
    """提取一个类别的视频特征；跳过有效的已有特征，遇到坏缓存时重新生成。"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- 开始处理类别: {category} (共 {len(video_list)} 个视频) ---")

    for index, video_path in enumerate(video_list, start=1):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{video_name}.pt")

        if os.path.exists(output_path):
            try:
                cached = torch.load(output_path, map_location="cpu", weights_only=True)
                if (
                    isinstance(cached, torch.Tensor)
                    and cached.ndim == 2
                    and cached.shape[1] == FEATURE_DIM
                    and cached.shape[0] > 0
                    and torch.isfinite(cached).all()
                ):
                    print(f"[{index}/{len(video_list)}] 跳过有效缓存: {video_name}")
                    continue
                print(f"[缓存无效] 将重新提取: {output_path}")
            except Exception as exc:
                print(f"[缓存读取失败] 将重新提取 {output_path}: {exc}")

        print(f"[{index}/{len(video_list)}] 正在提取: {video_name} ...", end="", flush=True)
        try:
            features = extract_features_from_video(video_path)
            if features is None:
                print(" 失败：未得到特征")
                continue

            # 先写临时文件，成功后再替换目标文件，避免留下半截缓存
            temp_path = output_path + ".tmp"
            torch.save(features, temp_path)
            os.replace(temp_path, output_path)
            print(f"；特征形状: {tuple(features.shape)}；已保存")
        except Exception as exc:
            print(f"\n[处理失败] {video_path}: {type(exc).__name__}: {exc}")
            temp_path = output_path + ".tmp"
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    print(f"--- 类别 {category} 处理结束 ---")


if __name__ == "__main__":
    if not os.path.isdir(UCF_DATASET_ROOT):
        raise FileNotFoundError(
            f"找不到 UCF-Crime 数据集根目录: {UCF_DATASET_ROOT}\n"
            "请检查路径，或设置环境变量 UCF_CRIME_ROOT。"
        )

    save_root = os.path.join(DATA_ROOT, "features")
    os.makedirs(save_root, exist_ok=True)

    normal_video_list, abnormal_video_list = collect_video_paths()
    if not normal_video_list and not abnormal_video_list:
        raise RuntimeError(
            "没有找到任何 MP4 视频。请检查 UCF_CRIME_ROOT 及其目录结构。"
        )

    process_video_list(
        normal_video_list, "normal", os.path.join(save_root, "normal")
    )
    process_video_list(
        abnormal_video_list, "abnormal", os.path.join(save_root, "abnormal")
    )

    print("\n==== 所有视频特征提取完毕！====")
