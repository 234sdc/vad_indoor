import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DATA_ROOT = os.environ.get("VAD_DATA_ROOT", "/data/Wcy/code/vad_indoor/UCF_indoor")
FEATURE_ROOT = os.path.join(DATA_ROOT, "features", "normal")
MODEL_ROOT = os.path.join(DATA_ROOT, "models")
BEST_CKPT_PATH = os.path.join(MODEL_ROOT, "tcn_best.pth")

SEG_LEN = 128
BATCH_SIZE = 16
EPOCHS = 30
PATIENCE = 6
SEED = 42
NUM_WORKERS = 0
FORCE_RETRAIN = os.environ.get("FORCE_TCN_RETRAIN", "0") == "1"

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(MODEL_ROOT, exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        )

    def forward(self, x):
        out = self.conv(x)
        if self.padding:
            out = out[:, :, :-self.padding]
        return out


class TemporalConvNet(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=128, num_layers=4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            c_in = in_dim if i == 0 else hidden_dim
            layers.extend([
                CausalConv1d(c_in, hidden_dim, kernel_size=3, dilation=dilation),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Conv1d(hidden_dim, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        # 输入 (B, 768, T)，输出 (B, T)
        return self.head(self.network(x)).squeeze(1)


def load_feature(path):
    try:
        feature = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        feature = torch.load(path, map_location="cpu")

    if not isinstance(feature, torch.Tensor):
        raise ValueError("文件内容不是 Tensor")
    if feature.ndim != 2 or feature.shape[1] != 768:
        raise ValueError(f"特征形状应为 (T, 768)，实际为 {tuple(feature.shape)}")
    if feature.shape[0] < 1:
        raise ValueError("特征序列为空")
    if not torch.isfinite(feature).all():
        raise ValueError("特征包含 NaN 或 Inf")
    return feature.float().contiguous()


class NormalFeatureDataset(Dataset):
    """每个样本来自指定视频；训练时随机裁剪，验证时确定性裁剪。"""

    def __init__(self, feature_files, seg_len=SEG_LEN, samples_per_video=4, training=True):
        self.seg_len = seg_len
        self.training = training
        self.samples = []

        for path in feature_files:
            try:
                feat = load_feature(path)
                if feat.shape[0] < 32:
                    print(f"[跳过短序列] {path}: T={feat.shape[0]} < 32")
                    continue
                # 保存路径与长度，避免将所有特征常驻内存
                for sample_idx in range(samples_per_video):
                    self.samples.append((path, sample_idx, int(feat.shape[0])))
            except Exception as exc:
                print(f"[跳过无效特征] {path}: {exc}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, sample_idx, _ = self.samples[idx]
        feat = load_feature(path)
        T = feat.shape[0]

        if T >= self.seg_len:
            if self.training:
                start = random.randint(0, T - self.seg_len)
            else:
                # 验证集固定取中间片段，避免验证指标受随机裁剪影响
                start = max(0, (T - self.seg_len) // 2)
            segment = feat[start:start + self.seg_len]
        else:
            pad_count = self.seg_len - T
            segment = torch.cat([feat, feat[-1:].repeat(pad_count, 1)], dim=0)

        return segment.transpose(0, 1).contiguous()


def anomaly_regularization_loss(scores, lambda_smooth=0.2, lambda_l1=0.05):
    """原有的正常样本分数正则化目标；它本身不提供异常类别监督。"""
    loss_low_score = scores.square().mean()
    loss_smooth = (scores[:, 1:] - scores[:, :-1]).square().mean()
    loss_sparse = scores.abs().mean()
    return loss_low_score + lambda_smooth * loss_smooth + lambda_l1 * loss_sparse


def split_by_video(feature_files, val_ratio=0.15, seed=SEED):
    """先按原始视频文件划分，再生成片段，避免同一视频泄漏到两侧。"""
    files = sorted(feature_files)
    rng = random.Random(seed)
    rng.shuffle(files)

    if len(files) < 2:
        raise RuntimeError(
            f"有效正常视频只有 {len(files)} 个，至少需要 2 个视频才能按视频划分训练集/验证集。"
        )

    val_count = max(1, int(round(len(files) * val_ratio)))
    val_count = min(val_count, len(files) - 1)
    val_files = sorted(files[:val_count])
    train_files = sorted(files[val_count:])
    return train_files, val_files


def main():
    set_seed()
    print(f"使用设备: {device}")
    print(f"正常特征目录: {FEATURE_ROOT}")

    if os.path.exists(BEST_CKPT_PATH) and not FORCE_RETRAIN:
        print(f"检测到已有权重，跳过训练: {BEST_CKPT_PATH}")
        print("若要重新训练，请设置 FORCE_TCN_RETRAIN=1。")
        return

    feature_files = sorted(glob.glob(os.path.join(FEATURE_ROOT, "*.pt")))
    if not feature_files:
        raise FileNotFoundError(
            f"未找到正常视频特征文件: {FEATURE_ROOT}/*.pt\n"
            "请先运行 dinov2_extract.py，并确认 DATA_ROOT 路径一致。"
        )

    train_files, val_files = split_by_video(feature_files)
    print(
        f"正常视频总数: {len(feature_files)} | "
        f"训练视频: {len(train_files)} | 验证视频: {len(val_files)}"
    )

    train_dataset = NormalFeatureDataset(train_files, training=True)
    val_dataset = NormalFeatureDataset(val_files, samples_per_video=1, training=False)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(
            f"数据集为空：train segments={len(train_dataset)}, "
            f"val segments={len(val_dataset)}。请检查特征文件和序列长度。"
        )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=False, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        drop_last=False, num_workers=NUM_WORKERS
    )

    model = TemporalConvNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    best_val_loss = float("inf")
    stop_counter = 0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_score_sum = 0.0
        train_batches = 0

        for batch_x in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(batch_x)
            loss = anomaly_regularization_loss(scores)
            if not torch.isfinite(loss):
                raise RuntimeError(f"训练损失出现 NaN/Inf，第 {epoch} 轮")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_score_sum += scores.detach().mean().item()
            train_batches += 1

        if train_batches == 0:
            raise RuntimeError("训练 DataLoader 没有产生 batch。")

        model.eval()
        val_loss_sum = 0.0
        val_score_sum = 0.0
        val_batches = 0
        with torch.inference_mode():
            for batch_x in val_loader:
                scores = model(batch_x.to(device))
                val_loss = anomaly_regularization_loss(scores)
                if not torch.isfinite(val_loss):
                    raise RuntimeError(f"验证损失出现 NaN/Inf，第 {epoch} 轮")
                val_loss_sum += val_loss.item()
                val_score_sum += scores.mean().item()
                val_batches += 1

        if val_batches == 0:
            raise RuntimeError("验证 DataLoader 没有产生 batch。")

        avg_train_loss = train_loss_sum / train_batches
        avg_train_score = train_score_sum / train_batches
        avg_val_loss = val_loss_sum / val_batches
        avg_val_score = val_score_sum / val_batches
        scheduler.step()

        print(
            f"Epoch [{epoch:02d}/{EPOCHS:02d}] "
            f"TrainLoss={avg_train_loss:.6f} ValLoss={avg_val_loss:.6f} "
            f"TrainScoreMean={avg_train_score:.5f} ValScoreMean={avg_val_score:.5f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            stop_counter = 0
            # 保持纯 state_dict 格式，兼容现有推理脚本的 load_state_dict()
            temp_path = BEST_CKPT_PATH + ".tmp"
            torch.save(model.state_dict(), temp_path)
            os.replace(temp_path, BEST_CKPT_PATH)
            print(f"已保存当前最优权重: {BEST_CKPT_PATH}")
        else:
            stop_counter += 1
            print(f"验证损失未改善：{stop_counter}/{PATIENCE}")
            if stop_counter >= PATIENCE:
                print("触发早停。")
                break

    print(f"训练结束；最佳轮次={best_epoch}，最佳验证损失={best_val_loss:.6f}")
    print("重要提示：该训练目标只对正常视频分数施加低分/平滑正则，不等于经过验证的异常概率。")
    print("请在独立正常/异常测试集上评估分数分布、AUROC/AUPRC、召回率与误报率。")


if __name__ == "__main__":
    main()
