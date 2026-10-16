import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

DATA_ROOT = "/data/Wcy/code/vad_indoor/UCF_indoor"
BEST_CKPT_PATH = os.path.join(DATA_ROOT, "models", "tcn_best.pth")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(os.path.join(DATA_ROOT, "models"), exist_ok=True)

# ====================新增：权重存在直接跳过训练====================
if os.path.exists(BEST_CKPT_PATH):
    print(f"✅ TCN最优权重 {BEST_CKPT_PATH} 已存在，跳过TCN训练！")
    exit(0)
# =================================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    def forward(self, x):
        out = self.conv(x)
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalConvNet(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=128, num_layers=4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            c_in = in_dim if i == 0 else hidden_dim
            layers.append(CausalConv1d(c_in, hidden_dim, kernel_size=3, dilation=dilation))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.network(x)
        score = self.head(feat)
        return score.squeeze(1)


class NormalFeatureDataset(Dataset):
    def __init__(self, feature_files, seg_len=128):
        self.seg_len = seg_len
        self.feat_list = []
        for fp in feature_files:
            ft = torch.load(fp, weights_only=True)
            if ft.shape[0] >= 32:
                self.feat_list.append(ft)

    def __len__(self):
        return len(self.feat_list) * 4

    def __getitem__(self, idx):
        feat = self.feat_list[idx % len(self.feat_list)]
        T = feat.shape[0]
        if T <= self.seg_len:
            pad = feat[-1:].repeat(self.seg_len - T, 1)
            seg = torch.cat([feat, pad], dim=0)
        else:
            st = np.random.randint(0, T - self.seg_len + 1)
            seg = feat[st:st + self.seg_len]
        return seg.transpose(0, 1)


if __name__ == "__main__":
    normal_pt_files = glob.glob(os.path.join(DATA_ROOT, "features/normal/*.pt"))
    print(f"加载正常样本数量：{len(normal_pt_files)}")
    full_dataset = NormalFeatureDataset(normal_pt_files, seg_len=128)

    # 划分训练集/验证集，用于早停
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=True)

    model = TemporalConvNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)

    epochs = 30
    lambda1 = 0.2
    lambda2 = 0.05
    patience = 6
    best_val_loss = float("inf")
    stop_counter = 0

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        total_train_loss = 0.0
        score_mean_train = 0.0
        for batch_x in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            S_t = model(batch_x)
            loss_s0 = torch.mean(torch.square(S_t))
            loss_smooth = lambda1 * torch.mean(torch.square(S_t[:, 1:] - S_t[:, :-1]))
            loss_l1 = lambda2 * torch.mean(torch.abs(S_t))
            loss = loss_s0 + loss_smooth + loss_l1
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            score_mean_train += S_t.mean().item()
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_score = score_mean_train / len(train_loader)

        # validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_x in val_loader:
                batch_x = batch_x.to(device)
                S_t = model(batch_x)
                loss_s0 = torch.mean(torch.square(S_t))
                loss_smooth = lambda1 * torch.mean(torch.square(S_t[:, 1:] - S_t[:, :-1]))
                loss_l1 = lambda2 * torch.mean(torch.abs(S_t))
                val_loss = loss_s0 + loss_smooth + loss_l1
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        scheduler.step()

        print(f"Epoch [{epoch:02d}/{epochs:02d}] TrainLoss:{avg_train_loss:.6f} ValLoss:{avg_val_loss:.6f} | AvgTrainScore:{avg_train_score:.4f}")

        # 早停+保存最优权重
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stop_counter = 0
            torch.save(model.state_dict(), BEST_CKPT_PATH)
            print(f"保存至 {BEST_CKPT_PATH}")
        else:
            stop_counter += 1
            print(f"⏳ 验证集未下降，早停计数 {stop_counter}/{patience}")
            if stop_counter >= patience:
                print(f"🛑 触发早停，连续{patience}轮验证损失不下降，终止训练！")
                break

    print(f"\nTCN训练结束！最优权重保存在：{BEST_CKPT_PATH}")
    print("💡说明：模型输出S_t，S_t越大代表异常概率越高，后续tcn_candidate_cut.py直接使用S_t作为异常分数S_coarse")
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

DATA_ROOT = "/data/Wcy/code/vad_indoor/UCF_indoor"
BEST_CKPT_PATH = os.path.join(DATA_ROOT, "models", "tcn_best.pth")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(os.path.join(DATA_ROOT, "models"), exist_ok=True)

# ====================新增：权重存在直接跳过训练====================
if os.path.exists(BEST_CKPT_PATH):
    print(f"✅ TCN最优权重 {BEST_CKPT_PATH} 已存在，跳过TCN训练！")
    exit(0)
# =================================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    def forward(self, x):
        out = self.conv(x)
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalConvNet(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=128, num_layers=4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            c_in = in_dim if i == 0 else hidden_dim
            layers.append(CausalConv1d(c_in, hidden_dim, kernel_size=3, dilation=dilation))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.network(x)
        score = self.head(feat)
        return score.squeeze(1)


class NormalFeatureDataset(Dataset):
    def __init__(self, feature_files, seg_len=128):
        self.seg_len = seg_len
        self.feat_list = []
        for fp in feature_files:
            ft = torch.load(fp, weights_only=True)
            if ft.shape[0] >= 32:
                self.feat_list.append(ft)

    def __len__(self):
        return len(self.feat_list) * 4

    def __getitem__(self, idx):
        feat = self.feat_list[idx % len(self.feat_list)]
        T = feat.shape[0]
        if T <= self.seg_len:
            pad = feat[-1:].repeat(self.seg_len - T, 1)
            seg = torch.cat([feat, pad], dim=0)
        else:
            st = np.random.randint(0, T - self.seg_len + 1)
            seg = feat[st:st + self.seg_len]
        return seg.transpose(0, 1)


if __name__ == "__main__":
    normal_pt_files = glob.glob(os.path.join(DATA_ROOT, "features/normal/*.pt"))
    print(f"加载正常样本数量：{len(normal_pt_files)}")
    full_dataset = NormalFeatureDataset(normal_pt_files, seg_len=128)

    # 划分训练集/验证集，用于早停
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=True)

    model = TemporalConvNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)

    epochs = 30
    lambda1 = 0.2
    lambda2 = 0.05
    patience = 6
    best_val_loss = float("inf")
    stop_counter = 0

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        total_train_loss = 0.0
        score_mean_train = 0.0
        for batch_x in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            S_t = model(batch_x)
            loss_s0 = torch.mean(torch.square(S_t))
            loss_smooth = lambda1 * torch.mean(torch.square(S_t[:, 1:] - S_t[:, :-1]))
            loss_l1 = lambda2 * torch.mean(torch.abs(S_t))
            loss = loss_s0 + loss_smooth + loss_l1
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            score_mean_train += S_t.mean().item()
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_score = score_mean_train / len(train_loader)

        # validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_x in val_loader:
                batch_x = batch_x.to(device)
                S_t = model(batch_x)
                loss_s0 = torch.mean(torch.square(S_t))
                loss_smooth = lambda1 * torch.mean(torch.square(S_t[:, 1:] - S_t[:, :-1]))
                loss_l1 = lambda2 * torch.mean(torch.abs(S_t))
                val_loss = loss_s0 + loss_smooth + loss_l1
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        scheduler.step()

        print(f"Epoch [{epoch:02d}/{epochs:02d}] TrainLoss:{avg_train_loss:.6f} ValLoss:{avg_val_loss:.6f} | AvgTrainScore:{avg_train_score:.4f}")

        # 早停+保存最优权重
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stop_counter = 0
            torch.save(model.state_dict(), BEST_CKPT_PATH)
            print(f"保存至 {BEST_CKPT_PATH}")
        else:
            stop_counter += 1
            print(f"⏳ 验证集未下降，早停计数 {stop_counter}/{patience}")
            if stop_counter >= patience:
                print(f"🛑 触发早停，连续{patience}轮验证损失不下降，终止训练！")
                break

    print(f"\nTCN训练结束！最优权重保存在：{BEST_CKPT_PATH}")
    print("💡说明：模型输出S_t，S_t越大代表异常概率越高，后续tcn_candidate_cut.py直接使用S_t作为异常分数S_coarse")
