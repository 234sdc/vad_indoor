import os, glob, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DATA_ROOT = os.environ.get("VAD_DATA_ROOT", "/data/Wcy/code/vad_indoor/UCF_indoor")
FEATURE_ROOT = os.path.join(DATA_ROOT, "features", "normal")
MODEL_ROOT = os.path.join(DATA_ROOT, "models")
CKPT = os.path.join(MODEL_ROOT, "tcn_best.pth")
CALIBRATION = os.path.join(MODEL_ROOT, "tcn_score_thresholds.json")
SEG_LEN, BATCH_SIZE, EPOCHS, PATIENCE, SEED = 128, 16, 40, 7, 42
FORCE_RETRAIN = os.environ.get("FORCE_TCN_RETRAIN", "0") == "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(MODEL_ROOT, exist_ok=True)

def set_seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def load_feature(path):
    try: x = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError: x = torch.load(path, map_location="cpu")
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != 768 or x.shape[0] < 2:
        raise ValueError(f"特征必须是非空 (T,768) Tensor，实际为 {getattr(x, 'shape', None)}")
    if not torch.isfinite(x).all(): raise ValueError("特征含 NaN/Inf")
    return x.float().contiguous()

class CausalConv1d(nn.Module):
    def __init__(self, a, b, kernel_size=3, dilation=1):
        super().__init__(); self.padding=(kernel_size-1)*dilation
        self.conv=nn.Conv1d(a,b,kernel_size,dilation=dilation,padding=self.padding)
    def forward(self,x):
        y=self.conv(x)
        return y[:,:,:-self.padding] if self.padding else y

class TemporalConvNet(nn.Module):
    # 输入历史特征，输出下一时刻的 768 维特征预测值
    def __init__(self,in_dim=768,hidden_dim=128,num_layers=4):
        super().__init__(); layers=[]
        for i in range(num_layers):
            layers += [CausalConv1d(in_dim if i==0 else hidden_dim,hidden_dim,3,2**i),
                       nn.BatchNorm1d(hidden_dim),nn.ReLU(),nn.Dropout(0.2)]
        self.network=nn.Sequential(*layers); self.head=nn.Conv1d(hidden_dim,in_dim,1)
    def forward(self,x): return self.head(self.network(x))

class NormalDataset(Dataset):
    def __init__(self, files, training, samples=4):
        self.items=[]; self.training=training; self.samples=samples
        for p in files:
            try:
                x=load_feature(p)
                for i in range(samples): self.items.append((p,i,int(x.shape[0])))
            except Exception as e: print(f"[跳过无效特征] {p}: {e}")
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        p,k,T=self.items[i]; x=load_feature(p)
        if T>=SEG_LEN:
            m=T-SEG_LEN
            start=random.randint(0,m) if self.training else (m//2 if self.samples==1 else round(k*m/(self.samples-1)))
            x=x[start:start+SEG_LEN]
        else: x=torch.cat((x,x[-1:].repeat(SEG_LEN-T,1)),0)
        # x[t] 预测 x[t+1]，不把目标帧送进预测器
        return x[:-1].T.contiguous(),x[1:].T.contiguous()

def split_by_video(files):
    if len(files)<2: raise RuntimeError(f"有效正常视频只有 {len(files)} 个，至少需要 2 个")
    files=sorted(files); random.Random(SEED).shuffle(files)
    n=max(1,min(len(files)-1,round(len(files)*0.15)))
    return sorted(files[n:]),sorted(files[:n])

def errors(pred,target): return (pred-target).square().mean(dim=1)

def main():
    set_seed(); print("设备:",device,"正常特征目录:",FEATURE_ROOT)
    if os.path.isfile(CKPT) and not FORCE_RETRAIN:
        print(f"已有模型，跳过训练：{CKPT}；重训请设置 FORCE_TCN_RETRAIN=1"); return
    files=glob.glob(os.path.join(FEATURE_ROOT,"*.pt"))
    if not files: raise FileNotFoundError(f"未找到正常特征：{FEATURE_ROOT}/*.pt")
    train_files,val_files=split_by_video(files)
    print(f"视频总数={len(files)} 训练={len(train_files)} 验证={len(val_files)}")
    tr=NormalDataset(train_files,True,4); va=NormalDataset(val_files,False,4)
    if not len(tr) or not len(va): raise RuntimeError("训练或验证数据集为空")
    tl=DataLoader(tr,batch_size=BATCH_SIZE,shuffle=True,drop_last=False,num_workers=0)
    vl=DataLoader(va,batch_size=BATCH_SIZE,shuffle=False,drop_last=False,num_workers=0)
    model=TemporalConvNet().to(device)
    opt=optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=1e-5)
    best=float("inf"); best_state=None; best_epoch=0; stale=0
    for ep in range(1,EPOCHS+1):
        model.train(); trsum=0; nb=0
        for x,y in tl:
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True)
            pred=model(x); loss=nn.functional.mse_loss(pred,y)
            if not torch.isfinite(loss): raise RuntimeError(f"训练损失 NaN/Inf: epoch {ep}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            trsum+=loss.item(); nb+=1
        model.eval(); vsum=0; nv=0
        with torch.inference_mode():
            for x,y in vl:
                x,y=x.to(device),y.to(device); pred=model(x); loss=nn.functional.mse_loss(pred,y)
                if not torch.isfinite(loss): raise RuntimeError(f"验证损失 NaN/Inf: epoch {ep}")
                vsum+=loss.item(); nv+=1
        if not nb or not nv: raise RuntimeError("DataLoader 没有产生 batch")
        v=vsum/nv; sched.step()
        print(f"Epoch [{ep:02d}/{EPOCHS}] TrainMSE={trsum/nb:.7f} ValMSE={v:.7f}")
        if v<best:
            best=v; best_epoch=ep; stale=0
            best_state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
        else:
            stale+=1; print(f"验证损失未改善 {stale}/{PATIENCE}")
            if stale>=PATIENCE: print("触发早停"); break
    if best_state is None: raise RuntimeError("未能得到有效模型")
    model.load_state_dict(best_state); model.to(device).eval()
    all_scores=[]
    with torch.inference_mode():
        for x,y in vl:
            pred=model(x.to(device)); all_scores.append(errors(pred,y.to(device)).cpu().numpy().ravel())
    scores=np.concatenate(all_scores).astype(np.float64); scores=scores[np.isfinite(scores)]
    if not scores.size: raise RuntimeError("验证误差为空，无法校准阈值")
    high=float(np.percentile(scores,99.5)); low=float(np.percentile(scores,99.0))
    tmp=CKPT+".tmp"; torch.save(best_state,tmp); os.replace(tmp,CKPT)
    cfg={"score_type":"per_timestep_mean_squared_feature_prediction_error","high_threshold":high,
         "low_threshold":min(low,high),"high_quantile_percent":99.5,"low_quantile_percent":99.0,
         "validation_error_count":int(scores.size),"best_epoch":best_epoch,"best_val_mse":best,"seed":SEED,
         "note":"Thresholds calibrated on normal validation videos only; evaluate on a separate labeled test set."}
    tmp=CALIBRATION+".tmp"
    with open(tmp,"w",encoding="utf-8") as f: json.dump(cfg,f,ensure_ascii=False,indent=2)
    os.replace(tmp,CALIBRATION)
    print(f"最佳轮次={best_epoch} ValMSE={best:.7f}")
    print("模型:",CKPT); print("阈值:",CALIBRATION)
    print(f"low={cfg['low_threshold']:.8f}, high={high:.8f}")
    print("注意：正常集分位数仅用于初步控制误报，不能代替独立异常测试集评估。")

if __name__=="__main__": main()
