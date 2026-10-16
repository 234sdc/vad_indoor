import os
import sys
import json
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import outlines
from outlines import generate, models

DATA_ROOT = "/data/Wcy/code/vad_indoor/UCF_indoor"
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(os.path.join(DATA_ROOT,"temp_clips"), exist_ok=True)

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = (kernel_size -1)*dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
    def forward(self, x):
        out = self.conv(x)
        return out[:,:,:-self.padding] if self.padding !=0 else out

class TemporalConvNet(nn.Module):
    def __init__(self,in_dim=768,hidden_dim=128,num_layers=4):
        super().__init__()
        layers=[]
        for i in range(num_layers):
            d=2**i
            cin = in_dim if i==0 else hidden_dim
            layers.append(CausalConv1d(cin,hidden_dim,3,d))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Conv1d(hidden_dim,1,1),nn.Sigmoid())
    def forward(self,x):
        return self.head(self.network(x)).squeeze(1)

dinov2 = torch.hub.load('facebookresearch/dinov2','dinov2_vitb14').to(device).eval()
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def extract_dinov2_features(video_path, target_fps=4):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1,int(round(fps/target_fps)))
    frames=[]
    idx=0
    while cap.isOpened():
        ret,frm = cap.read()
        if not ret:break
        if idx%interval ==0:
            rgb = cv2.cvtColor(frm,cv2.COLOR_BGR2RGB)
            frames.append(transform(rgb))
        idx +=1
    cap.release()
    if not frames:
        return None, fps
    ft = torch.stack(frames).to(device)
    out=[]
    with torch.no_grad():
        for i in range(0,len(ft),32):
            b = ft[i:i+32]
            f = dinov2(b)
            f = torch.nn.functional.normalize(f,p=2,dim=-1)
            out.append(f)
    return torch.cat(out,dim=0), fps

def gaussian_smooth(scores, sigma=1.5, window_size=5):
    radius = window_size//2
    x = np.arange(-radius, radius+1)
    kernel = np.exp(-0.5*(x/sigma)**2)
    kernel /= kernel.sum()
    padded = np.pad(scores,(radius,radius),mode="edge")
    return np.convolve(padded,kernel,mode="valid")

def detect_intervals(scores, tau_high=0.20, tau_low=0.16):
    intervals = []
    in_anomaly=False
    st=0
    for t,s in enumerate(scores):
        if not in_anomaly and s>=tau_high:
            in_anomaly=True
            st = t
        elif in_anomaly and s < tau_low:
            in_anomaly=False
            intervals.append((st,t))
    if in_anomaly:
        intervals.append((st, len(scores)-1))
    return intervals

def crop_clip(video_path, out_path, s_sec, e_sec, fps):
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(3)); h=int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w,h))
    sf = int(s_sec*fps)
    ef = int(e_sec*fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
    cur = sf
    while cap.isOpened() and cur <= ef:
        ret,frm = cap.read()
        if not ret:break
        out.write(frm)
        cur +=1
    cap.release(); out.release()

qwen_model = None
qwen_processor = None
qwen_outlines_model = None

def get_qwen():
    global qwen_model,qwen_processor,qwen_outlines_model
    if qwen_model is None:
        print("\n>>> [时序粗筛命中！按需唤醒 Qwen2‑VL‑7B‑Instruct + LoRA适配器] <<<")
        base_id = "Qwen/Qwen2-VL-7B-Instruct"
        lora_dir = os.path.join(DATA_ROOT,"models","qwen2vl7b_lora")
        qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        from peft import PeftModel
        qwen_model = PeftModel.from_pretrained(qwen_model, lora_dir)
        qwen_processor = AutoProcessor.from_pretrained(base_id)
        qwen_outlines_model = models.Transformers(qwen_model, qwen_processor.tokenizer)
    return qwen_model, qwen_processor, qwen_outlines_model

VLM_JSON_SCHEMA = '''
{
    "type":"object",
    "properties":{
        "is_anomaly":{"type":"boolean"},
        "anomaly_category":{"type":"string"},
        "confidence":{"type":"number","minimum":0.0,"maximum":1.0},
        "description":{"type":"string"}
    },
    "required":["is_anomaly","anomaly_category","confidence","description"]
}
'''

def analyze_video(video_path, alpha=0.3, tau_alert=0.5):
    print(f"\n==================================================")
    print(f"开始分析监控视频: {video_path}")
    print(f"==================================================")
    t0 = time.time()
    feat, orig_fps = extract_dinov2_features(video_path, target_fps=4)
    if feat is None:
        print("错误：视频读取失败")
        return
    tcn_weight = os.path.join(DATA_ROOT,"models","tcn_checkpoint.pth")
    tcn_model = TemporalConvNet().to(device)
    tcn_model.load_state_dict(torch.load(tcn_weight, weights_only=True))
    tcn_model.eval()

    with torch.no_grad():
        x = feat.unsqueeze(0).transpose(1,2)
        raw_scores = tcn_model(x).squeeze(0).cpu().numpy()
        smoothed = gaussian_smooth(raw_scores)
        intervals = detect_intervals(smoothed)
    print(f"扫描完成 | raw_max={raw_scores.max():.3f} smooth_max={smoothed.max():.3f}")
    if len(intervals)==0:
        print("\n>>>研判结论：安全放行，无异常，不唤醒大模型")
        return

    model, processor, ol_model = get_qwen()
    for idx,(s_idx,e_idx) in enumerate(intervals):
        peak_idx = s_idx + np.argmax(smoothed[s_idx:e_idx+1])
        S_coarse = float(smoothed[peak_idx])
        s_sec = max(0.0, (s_idx/4.0)-1.0)
        e_sec = (e_idx/4.0)+1.0
        if e_sec-s_sec>8.0:
            peak_sec = peak_idx /4.0
            s_sec = max(0.0, peak_sec-3.0)
            e_sec = peak_sec+3.0
        clip_path = os.path.join(DATA_ROOT,"temp_clips",f"candidate_{idx}.mp4")
        crop_clip(video_path, clip_path, s_sec, e_sec, orig_fps)
        prompt_text = """你是安防监控专家，观察室内监控短视频，识别异常行为。严格输出JSON。"""
        messages = [{
            "role":"user",
            "content":[
                {"type":"video","video":clip_path,"fps":4.0,"max_pixels":360*420},
                {"type":"text","text":prompt_text}
            ]
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_inp, vid_inp = process_vision_info(messages)
        inputs = processor(text=[text], images=img_inp, videos=vid_inp, return_tensors="pt").to(device)

        gen_json = generate.json(ol_model, VLM_JSON_SCHEMA)(inputs.input_ids, max_new_tokens=256)
        vlm_res = gen_json[0]

        is_ano = vlm_res["is_anomaly"]
        C = vlm_res["anomaly_category"]
        S_fine = float(vlm_res["confidence"])
        indicator = 1.0 if (is_ano and C.lower()!="normal") else 0.0
        S_final = alpha * S_coarse + (1.0-alpha)*S_fine * indicator
        print("\n[协同决策结果]")
        print(f"S_coarse={S_coarse:.3f} S_fine={S_fine:.3f} 类别:{C} S_final={S_final:.3f} threshold={tau_alert}")
        if S_final >= tau_alert:
            work_order = {
                "工单流水号":f"ALARM‑{int(time.time())}",
                "视频源":os.path.basename(video_path),
                "异常时间区间":f"{s_sec:.1f}s ~ {e_sec:.1f}s",
                "事件定性":C,
                "风险等级":"HIGH" if S_final>=0.7 else "MEDIUM",
                "综合威胁得分(S_final)":round(S_final,3),
                "时序‑语义置信度":{"S_coarse":round(S_coarse,3),"S_fine":round(S_fine,3)},
                "研判依据":vlm_res["description"],
                "处置建议":"立即呼叫值班人员到达现场制止，联动云台摄像机追踪目标"
            }
            print("\n🚨🚨🚨触发安防告警工单🚨🚨🚨")
            print(json.dumps(work_order, indent=2, ensure_ascii=False))
        else:
            print("\n>>>否决警报：语义核验为正常活动，抑制虚警")
    print(f"\n总耗时 {time.time()-t0:.2f}s")

if __name__ == "__main__":
    # 默认测试Assault样例视频
    default_test_video = "/data/Wcy/UCF/OpenDataLab___UCF-Crime/raw/UCF-Crime/Anomaly-Detection-Dataset/Anomaly-Videos-Part-1/Assault/Assault001_x264.mp4"
    target = sys.argv[-1] if len(sys.argv)>1 else default_test_video
    analyze_video(target)
