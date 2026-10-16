import cv2
import torch
from torchvision import transforms

DATA_ROOT = "/data/Wcy/code/vad_indoor/UCF_indoor"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device:{device}")
model = torch.hub.load('facebookresearch/dinov2','dinov2_vitb14').to(device).eval()
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
video_path = "/data/Wcy/UCF/OpenDataLab___UCF-Crime/raw/UCF-Crime/Anomaly-Detection-Dataset/Anomaly-Videos-Part-1/Assault/Assault001_x264.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
sample_interval = max(1,int(round(fps/4)))
frames=[]
idx=0
while cap.isOpened():
    ret,frm = cap.read()
    if not ret:break
    if idx%sample_interval ==0:
        rgb = cv2.cvtColor(frm,cv2.COLOR_BGR2RGB)
        frames.append(transform(rgb))
    idx +=1
cap.release()
frames_tensor = torch.stack(frames).to(device)
print(f"采样帧数 {frames_tensor.shape[0]}")
feats=[]
with torch.no_grad():
    for i in range(0, len(frames_tensor),32):
        batch = frames_tensor[i:i+32]
        ft = model(batch)
        ft = torch.nn.functional.normalize(ft,p=2,dim=-1)
        feats.append(ft.cpu())
all_feat = torch.cat(feats,dim=0)
print(f"特征shape {all_feat.shape}")
torch.save(all_feat, os.path.join(DATA_ROOT,"test_demo.pt"))
print("测试特征保存完成")
