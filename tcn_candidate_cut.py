import os, glob, json
import cv2
import torch
import torch.nn as nn
import numpy as np

DATA_ROOT=os.environ.get("VAD_DATA_ROOT","/data/Wcy/code/vad_indoor/UCF_indoor")
ABNORMAL_ROOT=os.environ.get("UCF_CRIME_ROOT","/data/Wcy/UCF/OpenDataLab___UCF-Crime/raw/UCF-Crime/Anomaly-Detection-Dataset")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
manifest_path=os.path.join(DATA_ROOT,"candidate_clips_manifest.json")
model_path=os.path.join(DATA_ROOT,"models","tcn_best.pth")
calibration_path=os.path.join(DATA_ROOT,"models","tcn_score_thresholds.json")
clip_out_dir=os.path.join(DATA_ROOT,"candidate_clips")
FORCE_REBUILD=os.environ.get("FORCE_REBUILD_CANDIDATES","0")=="1"

class CausalConv1d(nn.Module):
    def __init__(self,a,b,kernel_size=3,dilation=1):
        super().__init__(); self.padding=(kernel_size-1)*dilation
        self.conv=nn.Conv1d(a,b,kernel_size,dilation=dilation,padding=self.padding)
    def forward(self,x):
        y=self.conv(x); return y[:,:,:-self.padding] if self.padding else y

class TemporalConvNet(nn.Module):
    def __init__(self,in_dim=768,hidden_dim=128,num_layers=4):
        super().__init__(); layers=[]
        for i in range(num_layers):
            layers += [CausalConv1d(in_dim if i==0 else hidden_dim,hidden_dim,3,2**i),
                       nn.BatchNorm1d(hidden_dim),nn.ReLU(),nn.Dropout(0.2)]
        self.network=nn.Sequential(*layers); self.head=nn.Conv1d(hidden_dim,in_dim,1)
    def forward(self,x): return self.head(self.network(x))

def load_tensor(path):
    try: x=torch.load(path,map_location="cpu",weights_only=True)
    except TypeError: x=torch.load(path,map_location="cpu")
    if not isinstance(x,torch.Tensor) or x.ndim!=2 or x.shape[1]!=768 or x.shape[0]<2:
        raise ValueError(f"特征必须是 (T,768) 且 T>=2，实际 {getattr(x,'shape',None)}")
    if not torch.isfinite(x).all(): raise ValueError("特征含 NaN/Inf")
    return x.float().contiguous()

def gaussian_smooth(scores,sigma=1.5,window_size=5):
    if len(scores)<2: return scores
    radius=window_size//2; x=np.arange(-radius,radius+1)
    kernel=np.exp(-0.5*(x/sigma)**2); kernel/=kernel.sum()
    return np.convolve(np.pad(scores,(radius,radius),mode="edge"),kernel,mode="valid")

def detect_intervals(scores,tau_high,tau_low):
    intervals=[]; active=False; start=0
    for t,s in enumerate(scores):
        if not active and s>=tau_high: active=True; start=t
        elif active and s<tau_low:
            intervals.append((start,max(start,t-1))); active=False
    if active: intervals.append((start,len(scores)-1))
    return intervals

def crop_video(video_path,out_path,start_sec,end_sec):
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened(): print("警告：无法打开视频",video_path); return False
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps<=0: fps=25.0
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w<=0 or h<=0: cap.release(); return False
    start=max(0,int(start_sec*fps)); end=max(start,int(end_sec*fps))
    out=cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    if not out.isOpened(): cap.release(); print("警告：无法创建",out_path); return False
    cap.set(cv2.CAP_PROP_POS_FRAMES,start); cur=start; written=0
    while cap.isOpened() and cur<=end:
        ok,frame=cap.read()
        if not ok: break
        out.write(frame); cur+=1; written+=1
    cap.release(); out.release()
    if not written:
        if os.path.exists(out_path): os.remove(out_path)
        return False
    return True

def find_ucf_abnormal_video(name):
    for part in ["Anomaly-Videos-Part-1","Anomaly-Videos-Part-2","Anomaly-Videos-Part-3","Anomaly-Videos-Part-4"]:
        root=os.path.join(ABNORMAL_ROOT,part)
        if not os.path.isdir(root): continue
        for sub in os.listdir(root):
            p=os.path.join(root,sub,f"{name}.mp4")
            if os.path.isfile(p): return p
    return None

def score_video(model,feat):
    # 输入截止 t 的特征预测 t+1；分数第 0 项对应原序列索引 1。
    x=feat[:-1].T.unsqueeze(0).to(device); target=feat[1:]
    with torch.inference_mode(): pred=model(x).squeeze(0).T.cpu()
    return (pred-target).square().mean(dim=1).numpy()

def main():
    if os.path.exists(manifest_path) and not FORCE_REBUILD:
        print(f"{manifest_path} 已存在，跳过推理。重建请设置 FORCE_REBUILD_CANDIDATES=1。"); return
    if not os.path.isfile(model_path): raise FileNotFoundError(f"缺少模型 {model_path}，请先训练 train_tcn.py")
    if not os.path.isfile(calibration_path): raise FileNotFoundError(f"缺少阈值文件 {calibration_path}，请先训练新版本 train_tcn.py")
    with open(calibration_path,encoding="utf-8") as f: cfg=json.load(f)
    if cfg.get("score_type")!="per_timestep_mean_squared_feature_prediction_error":
        raise ValueError("分数定义不匹配，请重新训练和校准")
    high=float(cfg["high_threshold"]); low=float(cfg["low_threshold"])
    if not np.isfinite(high) or not np.isfinite(low) or low>high: raise ValueError("异常分数阈值无效")
    model=TemporalConvNet().to(device)
    try: state=torch.load(model_path,map_location=device,weights_only=True)
    except TypeError: state=torch.load(model_path,map_location=device)
    model.load_state_dict(state); model.eval()
    os.makedirs(clip_out_dir,exist_ok=True)
    feats=sorted(glob.glob(os.path.join(DATA_ROOT,"features","abnormal","*.pt")))
    sample_fps=4.0 # 必须与特征提取时的采样帧率一致
    buffer_sec=1.0; records=[]
    print(f"待处理特征={len(feats)} low={low:.8f} high={high:.8f}")
    for path in feats:
        name=os.path.splitext(os.path.basename(path))[0]; raw=find_ucf_abnormal_video(name)
        if raw is None: print(f"警告：找不到原始视频 {name}.mp4"); continue
        try:
            feat=load_tensor(path); raw_scores=score_video(model,feat)
        except Exception as e: print(f"警告：处理 {path} 失败：{e}"); continue
        if not np.isfinite(raw_scores).all(): print(f"警告：分数无效 {name}"); continue
        smooth=gaussian_smooth(raw_scores); intervals=detect_intervals(smooth,high,low)
        print(f"[{name}] max_error={raw_scores.max():.8f} intervals={len(intervals)}")
        for j,(s,e) in enumerate(intervals):
            if e<s: continue
            peak=s+int(np.argmax(smooth[s:e+1])); peak_score=float(smooth[peak])
            # 分数索引 0 对应原特征序列索引 1
            start_sec=max(0.0,(s+1)/sample_fps-buffer_sec); end_sec=(e+1)/sample_fps+buffer_sec
            if end_sec-start_sec>8:
                peak_sec=(peak+1)/sample_fps; start_sec=max(0.0,peak_sec-3); end_sec=peak_sec+3
            clip_name=f"{name}_clip_{j:02d}.mp4"; dest=os.path.join(clip_out_dir,clip_name)
            if not crop_video(raw,dest,start_sec,end_sec): print("警告：切片失败",clip_name); continue
            records.append({"clip_name":clip_name,"source_video":f"{name}.mp4",
                "start_time_sec":round(start_sec,2),"end_time_sec":round(end_sec,2),
                "duration_sec":round(end_sec-start_sec,2),"S_coarse":round(peak_score,8),
                "score_type":"mean_squared_feature_prediction_error"})
            print(f"  -> {clip_name} {start_sec:.1f}-{end_sec:.1f}s error={peak_score:.8f}")
    tmp=manifest_path+".tmp"
    with open(tmp,"w",encoding="utf-8") as f: json.dump(records,f,indent=2,ensure_ascii=False)
    os.replace(tmp,manifest_path)
    print(f"完成：{len(records)} 个候选切片，清单 {manifest_path}")

if __name__=="__main__": main()
