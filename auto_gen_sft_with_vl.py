# auto_gen_sft_with_vl.py 【关键帧图片版本，生成jsonl存image路径，不再存video】
import os
# 【放在最前面，解决cuda显存碎片】
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import re
import torch
import cv2
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

DATA_ROOT = "/data/Wcy/code/vad_indoor/UCF_indoor"
manifest_path = os.path.join(DATA_ROOT, "candidate_clips_manifest_filtered.json")
out_jsonl = os.path.join(DATA_ROOT, "sft_dataset", "UCF_indoor_sft.jsonl")
cache_frame_dir = os.path.join(DATA_ROOT, "cache_frames")
os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
os.makedirs(cache_frame_dir, exist_ok=True)

# =====================新增：jsonl已存在直接跳过，适配一键脚本=====================
if os.path.exists(out_jsonl):
    print(f"✅ SFT数据集 {out_jsonl} 已经生成完成，直接跳过本步骤！")
    exit(0)
# ===============================================================================

# 前置校验：过滤后的清单是否存在
if not os.path.exists(manifest_path):
    print(f"❌ 错误：找不到 {manifest_path}，请先执行filter_low_score_clips.py")
    exit(1)

ucf_info_map = {
    "Abuse": {"anomaly_category": "Abuse虐待行为", "confidence":0.92},
    "Arrest": {"anomaly_category": "Arrest抓捕行动", "confidence":0.92},
    "Arson": {"anomaly_category": "Arson纵火", "confidence":0.94},
    "Assault": {"anomaly_category": "Assault打架斗殴", "confidence":0.92},
    "Burglary": {"anomaly_category": "Burglary入室盗窃", "confidence":0.90},
    "Explosion": {"anomaly_category": "Explosion爆炸", "confidence":0.94},
    "Fighting": {"anomaly_category": "Fighting聚众打斗", "confidence":0.93},
    "RoadAccidents": {"anomaly_category": "RoadAccidents交通事故", "confidence":0.91},
    "Robbery": {"anomaly_category": "Robbery抢劫", "confidence":0.92},
    "Shooting": {"anomaly_category": "Shooting枪击事件", "confidence":0.93},
    "Shoplifting": {"anomaly_category": "Shoplifting商店偷窃", "confidence":0.90},
    "Stealing": {"anomaly_category": "Stealing偷窃", "confidence":0.90},
    "Vandalism": {"anomaly_category": "Vandalism破坏公物", "confidence":0.91},
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print("加载Qwen2-VL-7B-Instruct，用于自动生成SFT数据集description描述...")
model_id = "/data/Wcy/models/Qwen2-VL-7B-Instruct"

# ========== 模型加载，增加梯度检查点省显存 ==========
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
    local_files_only=True,
    low_cpu_mem_usage=True
)
model.gradient_checkpointing_enable()
processor = AutoProcessor.from_pretrained(
    model_id,
    local_files_only=True
)

desc_prompt = """你是安防视频标注助手，观察这个监控短视频。
只输出简短中文行为描述，不要输出JSON，不要解释，不要多余话，1-2句话说明视频里面发生什么事件。
"""
prompt_template_sft = """你是一个智能安防监控专家。请仔细观察这几张监控截图，判断是否存在异常行为（如打架斗殴、人员摔倒、盗窃、设施破坏等）。
严格输出JSON格式：
{
  "is_anomaly": true/false,
  "anomaly_category": "类别名称，无异常填Normal",
  "confidence":0.0~1.0浮点数,
  "description":"简要描述画面行为证据"
}
"""
# generation加速配置，greedy解码，速度最快
gen_kwargs = {
    "max_new_tokens":128,
    "do_sample":False,
    "num_beams":1,
}

# 提取视频关键帧，保存图片到cache，返回图片绝对路径列表
def extract_keyframes_save(video_path, clip_name):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
    indices = [0, total_frames//2]
    img_paths = []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            img_save_name = f"{clip_name}_frame_{i}.jpg"
            img_full_path = os.path.join(cache_frame_dir, img_save_name)
            pil_img.save(img_full_path)
            img_paths.append(img_full_path)
    cap.release()
    return img_paths

# 加载清单
with open(manifest_path, "r", encoding="utf-8") as f:
    clip_items = json.load(f)
sft_samples = []
save_interval = 20 # 每处理20条，保存一次，断点保护
# 正则提取类别前缀：提取字母开头，直到第一个数字
pattern = re.compile(r"^([A-Za-z]+)")

for idx,item in enumerate(clip_items):
    clip_name = item["clip_name"]
    clip_full_path = os.path.join(DATA_ROOT, "candidate_clips", clip_name)
    match = pattern.match(clip_name)
    if not match:
        print(f"跳过无法提取类别：{clip_name}")
        continue
    prefix = match.group(1)
    info = ucf_info_map.get(prefix)
    if info is None:
        print(f"跳过未知类别 {clip_name}")
        continue
    if not os.path.exists(clip_full_path):
        print(f"警告：视频文件不存在 {clip_full_path}，跳过")
        continue
    print(f"\n[{idx+1}/{len(clip_items)}] 处理 {clip_name}")
    try:
        # 1. 提取并保存关键帧图片
        img_path_list = extract_keyframes_save(clip_full_path, clip_name)
        if len(img_path_list) == 0:
            print(f"⚠️ {clip_name} 提取关键帧失败，跳过")
            continue

        # 2. 调用Qwen2-VL生成描述（仍然使用video输入生成描述，这一步只做标注，不进SFT训练集）
        messages_desc = [
            {
                "role":"user",
                "content":[
                    {"type":"video","video":clip_full_path,"fps":2.0,"max_pixels":512*512},
                    {"type":"text","text":desc_prompt}
                ]
            }
        ]
        text_q = processor.apply_chat_template(messages_desc, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(messages_desc)
        inputs = processor(text=[text_q], images=img_in, videos=vid_in, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(**inputs,**gen_kwargs)
        trimmed = [out[len(inp):] for inp,out in zip(inputs.input_ids, gen_ids)]
        desc_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

        ans_dict = {
            "is_anomaly": True,
            "anomaly_category": info["anomaly_category"],
            "confidence": info["confidence"],
            "description": desc_text
        }

        # =========核心改动：SFT样本写入image路径，不再写video=========
        user_content = []
        for p in img_path_list:
            user_content.append({"type":"image", "image":p})
        user_content.append({"type":"text","text":prompt_template_sft})

        sample = {
            "messages":[
                {
                    "role":"user",
                    "content": user_content
                },
                {
                    "role":"assistant",
                    "content": json.dumps(ans_dict, ensure_ascii=False)
                }
            ]
        }
        sft_samples.append(json.dumps(sample, ensure_ascii=False))

        # 定期保存，防止崩溃丢失
        if (idx+1) % save_interval == 0:
            with open(out_jsonl,"w",encoding="utf-8") as fw:
                for line in sft_samples:
                    fw.write(line+"\n")
            print(f"💾 临时保存，已完成 {idx+1} 条")

        # 释放显存
        del inputs, gen_ids, trimmed
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ 处理 {clip_name} 出错，跳过，err: {str(e)}")
        torch.cuda.empty_cache()
        continue

# 最终写入全部
with open(out_jsonl,"w",encoding="utf-8") as fw:
    for line in sft_samples:
        fw.write(line+"\n")
print(f"\n✅全部处理完毕！有效样本数 {len(sft_samples)} 输出 {out_jsonl}")
print("⚠️人工轻量复核：快速浏览jsonl，删掉描述乱码、幻觉的行！")
print("##################################################################################")
print("⚠️ 【极轻量人工校验，**不需要播放视频！！**】")
print("打开文件 UCF_indoor/sft_dataset/UCF_indoor_sft.jsonl")
print("只做：快速浏览全部行，如果看到明显不合理/幻觉乱描述的条目直接整行删除，不需要修改任何字段内容！")
print("✅类别、置信度、行为描述全部由脚本+Qwen2‑VL自动生成，不用手动填写任何文字！")
print("##################################################################################")
print("浏览完毕，确认没有明显错误，按回车继续QLoRA微调...")
