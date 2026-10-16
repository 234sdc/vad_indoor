import os
import json

DATA_ROOT = "/data/Wcy/code/vad_indoor/UCF_indoor"
clip_manifest = os.path.join(DATA_ROOT, "candidate_clips_manifest.json")
sft_out_path = os.path.join(DATA_ROOT, "sft_dataset", "UCF_indoor_sft.jsonl")
os.makedirs(os.path.join(DATA_ROOT, "sft_dataset"), exist_ok=True)

prompt_template = """你是一个智能安防监控专家。请仔细观察这段室内监控短视频，判断是否存在异常行为（如打架斗殴、人员摔倒、盗窃、设施破坏等）。
严格输出JSON格式：
{
  "is_anomaly": true/false,
  "anomaly_category": "类别名称，无异常填Normal",
  "confidence":0.0~1.0浮点数,
  "description":"简要描述画面行为证据"
}
"""

if __name__ == "__main__":
    with open(clip_manifest, "r", encoding="utf-8") as f:
        clip_list = json.load(f)
    sft_samples = []
    for item in clip_list:
        clip_file = item["clip_name"]
        video_rel_path = os.path.join(DATA_ROOT, "candidate_clips", clip_file)
        sample = {
            "messages":[
                {
                    "role":"user",
                    "content":[
                        {"type":"video","video":video_rel_path,"fps":4.0,"max_pixels":360*420},
                        {"type":"text","text":prompt_template}
                    ]
                },
                {
                    "role":"assistant",
                    "content":'{"is_anomaly":true,"anomaly_category":"Unknown","confidence":0.90,"description":"待人工修正：描述视频内异常行为"}'
                }
            ]
        }
        sft_samples.append(sample)
    with open(sft_out_path, "w", encoding="utf-8") as fw:
        for s in sft_samples:
            fw.write(json.dumps(s, ensure_ascii=False)+"\n")
    print(f"SFT原始数据集生成完成，样本数：{len(sft_samples)}")
    print(f"输出文件：{sft_out_path}")
    print("⚠️重要：打开该jsonl，人工修正每一条assistant的content，修正anomaly_category和description！")
