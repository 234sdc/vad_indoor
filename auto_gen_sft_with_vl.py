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

DATA_ROOT = os.environ.get("VAD_DATA_ROOT", "/data/Wcy/code/vad_indoor/UCF_indoor")
manifest_path = os.path.join(DATA_ROOT, "candidate_clips_manifest_filtered.json")
out_jsonl = os.path.join(DATA_ROOT, "sft_dataset", "UCF_indoor_sft.jsonl")
cache_frame_dir = os.path.join(DATA_ROOT, "cache_frames")
os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
os.makedirs(cache_frame_dir, exist_ok=True)

# 已有数据集默认跳过；设置 FORCE_REBUILD_SFT=1 可重新生成
if os.path.exists(out_jsonl) and os.environ.get("FORCE_REBUILD_SFT", "0") != "1":
    print(f"✅ SFT数据集 {out_jsonl} 已经存在，跳过生成。")
    print("如需按新描述提示词重新生成，请设置 FORCE_REBUILD_SFT=1。")
    raise SystemExit(0)

if not os.path.exists(manifest_path):
    print(f"❌ 错误：找不到 {manifest_path}，请先执行 filter_low_score_clips.py")
    raise SystemExit(1)

ucf_info_map = {
    "Abuse": {"anomaly_category": "Abuse虐待行为", "confidence": 0.92},
    "Arrest": {"anomaly_category": "Arrest抓捕行动", "confidence": 0.92},
    "Arson": {"anomaly_category": "Arson纵火", "confidence": 0.94},
    "Assault": {"anomaly_category": "Assault打架斗殴", "confidence": 0.92},
    "Burglary": {"anomaly_category": "Burglary入室盗窃", "confidence": 0.90},
    "Explosion": {"anomaly_category": "Explosion爆炸", "confidence": 0.94},
    "Fighting": {"anomaly_category": "Fighting聚众打斗", "confidence": 0.93},
    "RoadAccidents": {"anomaly_category": "RoadAccidents交通事故", "confidence": 0.91},
    "Robbery": {"anomaly_category": "Robbery抢劫", "confidence": 0.92},
    "Shooting": {"anomaly_category": "Shooting枪击事件", "confidence": 0.93},
    "Shoplifting": {"anomaly_category": "Shoplifting商店偷窃", "confidence": 0.90},
    "Stealing": {"anomaly_category": "Stealing偷窃", "confidence": 0.90},
    "Vandalism": {"anomaly_category": "Vandalism破坏公物", "confidence": 0.91},
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print("加载Qwen2-VL-7B-Instruct，读取过滤后的候选清单并生成具体事件描述...")
model_id = "/data/Wcy/models/Qwen2-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
    local_files_only=True,
    low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)

# 要求描述可观察到的具体动作，而不是只复述类别或写“发生了异常”
desc_prompt = """你是一名严谨的安防视频事件标注员。请完整观察输入的监控短视频，并用中文具体描述画面里实际发生了什么。
要求：
1. 写2到4句话，尽量按事件发生顺序描述：有哪些人/车辆或物体、各自做了什么动作、动作如何发展、结果或现场变化是什么。
2. 优先描述可直接观察到的行为证据，例如推搡、挥拳、踢打、追逐、拿走物品、点火、撞击、破坏物品等；只有视频确实显示时才能写。
3. 不要只写“发生了打架/盗窃/破坏”等类别名称，也不要写“有人行为异常”这种空泛结论；要说明具体动作及其对象。
4. 不得臆测人物身份、动机、关系、物品价值、画面外事件或视频未显示的结果。
5. 如果关键帧/视频证据不足以确认具体动作，明确说“画面证据有限，无法确定具体行为”，并只描述看得见的内容。
只输出描述正文，不要标题、JSON、分析过程或额外解释。"""

prompt_template_sft = """你是一个严谨的智能安防监控分析助手。请根据提供的监控关键帧，判断画面中可观察到的行为，并具体说明发生了什么。
请严格输出JSON对象，字段如下：
{
  "is_anomaly": true或false,
  "anomaly_category": "具体类别；无法确认异常时填Normal或Uncertain",
  "confidence": 0.0到1.0之间的数值,
  "description": "用2到4句话描述可观察到的具体人物/物体、动作、动作顺序及结果；不要只复述类别，不要臆测画面外信息"
}
必须以画面证据为准。证据不足时应降低confidence，并在description中明确说明不确定性。"""

gen_kwargs = {"max_new_tokens": 160, "do_sample": False, "num_beams": 1}

def extract_keyframes_save(video_path, clip_name):
    """从片段中提取开头、中间和结尾附近的关键帧，返回已保存图片的绝对路径。"""
    cap = cv2.VideoCapture(video_path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return []
        # 三个时间位置比仅看两帧更有机会覆盖动作变化
        indices = sorted(set([0, total_frames // 2, max(0, total_frames - 1)]))
        img_paths = []
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(clip_name)[0])
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_path = os.path.join(cache_frame_dir, f"{safe_name}_frame_{i}.jpg")
            Image.fromarray(rgb).save(img_path, quality=92)
            img_paths.append(img_path)
        return img_paths
    finally:
        cap.release()

with open(manifest_path, "r", encoding="utf-8") as f:
    clip_items = json.load(f)
if not isinstance(clip_items, list):
    raise ValueError(f"候选清单格式错误，应为JSON数组：{manifest_path}")

sft_samples = []
save_interval = 20
pattern = re.compile(r"^([A-Za-z]+)")
processed_count = 0

for idx, item in enumerate(clip_items):
    clip_name = item.get("clip_name", "")
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
    if not os.path.isfile(clip_full_path):
        print(f"警告：视频文件不存在 {clip_full_path}，跳过")
        continue

    print(f"\n[{idx + 1}/{len(clip_items)}] 处理 {clip_name}")
    inputs = gen_ids = trimmed = None
    try:
        img_path_list = extract_keyframes_save(clip_full_path, clip_name)
        if not img_path_list:
            print(f"⚠️ {clip_name} 提取关键帧失败，跳过")
            continue

        # 用视频内容生成细粒度描述；类别前缀仅作为数据集来源信息，不替代视觉证据
        messages_desc = [{
            "role": "user",
            "content": [
                {"type": "video", "video": clip_full_path, "fps": 2.0, "max_pixels": 512 * 512},
                {"type": "text", "text": desc_prompt},
            ],
        }]
        text_q = processor.apply_chat_template(
            messages_desc, tokenize=False, add_generation_prompt=True
        )
        img_in, vid_in = process_vision_info(messages_desc)
        inputs = processor(
            text=[text_q], images=img_in, videos=vid_in, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            gen_ids = model.generate(**inputs, **gen_kwargs)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
        desc_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        desc_text = re.sub(r"\s+", " ", desc_text).strip().strip('"“”')

        if len(desc_text) < 8:
            desc_text = "画面证据有限，无法确定具体行为，建议人工复核该片段。"

        # 保留来源类别作为弱标签，但明确description必须来自视频可见证据。
        # 候选片段来自异常类别视频，不代表片段本身一定含异常，生成后仍需抽查。
        ans_dict = {
            "is_anomaly": True,
            "anomaly_category": info["anomaly_category"],
            "confidence": info["confidence"],
            "description": desc_text,
        }

        user_content = [{"type": "image", "image": p} for p in img_path_list]
        user_content.append({"type": "text", "text": prompt_template_sft})
        sample = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": json.dumps(ans_dict, ensure_ascii=False)},
            ]
        }
        sft_samples.append(json.dumps(sample, ensure_ascii=False))
        processed_count += 1

        if processed_count % save_interval == 0:
            with open(out_jsonl, "w", encoding="utf-8") as fw:
                fw.write("\n".join(sft_samples) + "\n")
            print(f"💾 临时保存，已完成 {processed_count} 条")

    except Exception as e:
        print(f"❌ 处理 {clip_name} 出错，跳过，err: {e}")
        continue
    finally:
        if inputs is not None:
            del inputs
        if gen_ids is not None:
            del gen_ids
        if trimmed is not None:
            del trimmed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

with open(out_jsonl, "w", encoding="utf-8") as fw:
    if sft_samples:
        fw.write("\n".join(sft_samples) + "\n")
print(f"\n✅处理完毕！有效样本数 {len(sft_samples)}，输出 {out_jsonl}")
print("注意：is_anomaly、anomaly_category和confidence目前仍来自数据集类别弱标签；异常来源视频中的候选片段不一定都包含异常。")
print("请重点复核description是否具体、是否与画面一致，并剔除标签或描述与实际画面不符的样本。")
