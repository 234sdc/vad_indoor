import os
import json
DATA_ROOT = "/data/Wcy/code/vad_indoor/UCF_indoor"
manifest_in = os.path.join(DATA_ROOT, "candidate_clips_manifest.json")
manifest_out = os.path.join(DATA_ROOT, "candidate_clips_manifest_filtered.json")
clip_dir = os.path.join(DATA_ROOT, "candidate_clips")
# 阈值：低于该分数判定为TCN虚警，直接丢弃，可根据训练情况微调
S_THRESHOLD = 0.0004

# =========新增：过滤结果存在则直接跳过，适配一键脚本=========
if os.path.exists(manifest_out):
    print(f"✅ 过滤后的清单 {manifest_out} 已存在，跳过低分过滤步骤！")
    exit(0)
# ============================================================

# 检查输入原始manifest是否存在
if not os.path.exists(manifest_in):
    print(f"❌ 错误：找不到输入文件 {manifest_in}，请先执行tcn_candidate.py生成候选切片")
    exit(1)

with open(manifest_in, "r", encoding="utf-8") as f:
    clip_list = json.load(f)
keep_list = []
drop_count = 0
for item in clip_list:
    score = item["S_coarse"]
    clip_path = os.path.join(clip_dir, item["clip_name"])
    if score >= S_THRESHOLD:
        keep_list.append(item)
    else:
        drop_count += 1
        if os.path.exists(clip_path):
            os.remove(clip_path)

with open(manifest_out, "w", encoding="utf-8") as fw:
    json.dump(keep_list, fw, indent=2, ensure_ascii=False)
print(f"原始切片总数：{len(clip_list)}")
print(f"S_coarse < {S_THRESHOLD}，丢弃虚警切片数量：{drop_count}")
print(f"保留用于SFT构建切片数量：{len(keep_list)}")
print(f"过滤后清单输出至：{manifest_out}")
