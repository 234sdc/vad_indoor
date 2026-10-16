import json
import math
import os
import shutil
import sys
from pathlib import Path

# VAD_DATA_ROOT 应指向 UCF_indoor 数据目录，而不是仓库根目录。
DATA_ROOT = Path(
    os.environ.get(
        "VAD_DATA_ROOT",
        "/data/Wcy/code/vad_indoor/UCF_indoor",
    )
)
MANIFEST_IN = DATA_ROOT / "candidate_clips_manifest.json"
MANIFEST_OUT = DATA_ROOT / "candidate_clips_manifest_filtered.json"
CLIP_DIR = DATA_ROOT / "candidate_clips"
THRESHOLD_FILE = DATA_ROOT / "models" / "tcn_score_thresholds.json"

# 默认使用本次 TCN 正常验证集标定得到的 high_threshold。
# 如需手动指定，可设置 S_THRESHOLD，例如 S_THRESHOLD=0.0014。
# S_coarse 是候选区间内的平均逐时刻预测误差；这是启发式过滤阈值，
# 最终应结合人工抽查/带标注测试集评估。
FORCE_FILTER = os.environ.get("FORCE_FILTER", "0") == "1"


def fail(message: str, code: int = 1) -> None:
    print(f"错误：{message}", file=sys.stderr)
    raise SystemExit(code)


def load_threshold() -> float:
    override = os.environ.get("S_THRESHOLD")
    if override:
        try:
            value = float(override)
        except ValueError:
            fail(f"S_THRESHOLD 必须是数字，当前值为 {override!r}")
        if not math.isfinite(value):
            fail("S_THRESHOLD 必须是有限数值")
        print(f"使用环境变量指定的过滤阈值：{value:.8f}")
        return value

    if not THRESHOLD_FILE.exists():
        fail(
            f"找不到阈值文件 {THRESHOLD_FILE}。"
            "请先运行 train_tcn.py 生成阈值文件，"
            "或通过环境变量 S_THRESHOLD 手动指定阈值。"
        )

    try:
        with THRESHOLD_FILE.open("r", encoding="utf-8") as f:
            config = json.load(f)
        value = float(config["high_threshold"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        fail(f"无法从 {THRESHOLD_FILE} 读取有效的 high_threshold")

    if not math.isfinite(value):
        fail(f"{THRESHOLD_FILE} 中的 high_threshold 不是有限数值")

    print(f"读取 TCN 标定阈值：{THRESHOLD_FILE}")
    print(f"使用 high_threshold={value:.8f}")
    return value


def main() -> None:
    if not MANIFEST_IN.exists():
        fail(f"找不到输入清单 {MANIFEST_IN}，请先运行 tcn_candidate_cut.py")

    if MANIFEST_OUT.exists() and not FORCE_FILTER:
        print(f"过滤后的清单已存在：{MANIFEST_OUT}")
        print("为避免误覆盖，本次跳过。需要重新过滤时运行：")
        print("  FORCE_FILTER=1 python filter_low_score_clips.py")
        print("也可以设置 S_THRESHOLD=数值 手动指定过滤阈值。")
        return

    try:
        with MANIFEST_IN.open("r", encoding="utf-8") as f:
            clip_list = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"读取输入清单失败：{exc}")

    if not isinstance(clip_list, list):
        fail("输入清单格式不正确：JSON 顶层应为数组")

    threshold = load_threshold()
    keep_list = []
    drop_count = 0
    missing_count = 0

    for index, item in enumerate(clip_list):
        if not isinstance(item, dict):
            fail(f"清单第 {index + 1} 项不是 JSON 对象")

        try:
            score = float(item["S_coarse"])
        except (KeyError, TypeError, ValueError):
            fail(f"清单第 {index + 1} 项缺少有效的 S_coarse 字段")

        if not math.isfinite(score):
            fail(f"清单第 {index + 1} 项的 S_coarse 不是有限数值")

        # 只过滤清单，不删除原始 MP4，避免阈值设置不合适时造成数据丢失。
        if score >= threshold:
            keep_list.append(item)
        else:
            drop_count += 1

        clip_name = item.get("clip_name")
        if isinstance(clip_name, str) and not (CLIP_DIR / clip_name).is_file():
            missing_count += 1

    # 若强制重跑，先备份旧清单；不触碰任何候选 MP4 文件。
    if MANIFEST_OUT.exists() and FORCE_FILTER:
        backup_path = MANIFEST_OUT.with_suffix(MANIFEST_OUT.suffix + ".bak")
        try:
            shutil.copy2(MANIFEST_OUT, backup_path)
            print(f"已备份旧过滤清单：{backup_path}")
        except OSError as exc:
            fail(f"备份旧过滤清单失败，为避免覆盖数据已停止：{exc}")

    try:
        temp_path = MANIFEST_OUT.with_suffix(MANIFEST_OUT.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(keep_list, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(temp_path, MANIFEST_OUT)
    except OSError as exc:
        fail(f"写入过滤清单失败：{exc}")

    print(f"原始候选切片数：{len(clip_list)}")
    print(f"过滤阈值：S_coarse >= {threshold:.8f}")
    print(f"低分过滤数量：{drop_count}")
    print(f"保留候选数量：{len(keep_list)}")
    if missing_count:
        print(f"警告：有 {missing_count} 个清单条目对应的 MP4 文件不存在于 {CLIP_DIR}")
    print(f"过滤后清单：{MANIFEST_OUT}")
    print("注意：本脚本只更新过滤清单，不会删除候选 MP4 文件。")


if __name__ == "__main__":
    main()
