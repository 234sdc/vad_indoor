#!/bin/bash
set -e
PROJECT_ROOT="/data/Wcy/code/vad_indoor"
cd ${PROJECT_ROOT}

echo "===== 步骤1：DINOv2批量提取视频特征 ====="
# dinov2_extract.py内部自带单视频跳过逻辑
python dinov2_extract.py

echo -e "\n===== 步骤2：训练TCN时序粗筛网络 ====="
TCN_CKPT="${PROJECT_ROOT}/UCF_indoor/models/tcn_best.pth"
if [ -f "$TCN_CKPT" ];then
    echo "✅ TCN模型权重 tcn_best.pth 已存在，跳过TCN训练"
else
    python train_tcn.py
fi

echo -e "\n===== 步骤3：TCN推理，生成候选切片和原始manifest ====="
RAW_MANIFEST="${PROJECT_ROOT}/UCF_indoor/candidate_clips_manifest.json"
if [ -f "$RAW_MANIFEST" ];then
    echo "✅ candidate_clips_manifest.json 已存在，跳过TCN推理+视频切片"
else
    python tcn_candidate_cut.py
fi

echo -e "\n===== 步骤4：脚本自动过滤低分虚警切片，减少样本量【无需看视频】 ====="
FILTER_MANIFEST="${PROJECT_ROOT}/UCF_indoor/candidate_clips_manifest_filtered.json"
if [ -f "$FILTER_MANIFEST" ];then
    echo "✅ candidate_clips_manifest_filtered.json 已存在，跳过过滤"
else
    python filter_low_score_clips.py
fi

echo -e "\n===== 步骤5：Qwen2‑VL大模型辅助生成SFT数据集，自动生成类别、confidence、视频行为description描述【无需人工写文本】 ====="
SFT_JSONL="${PROJECT_ROOT}/UCF_indoor/sft_dataset/UCF_indoor_sft.jsonl"
if [ -f "$SFT_JSONL" ];then
    echo "✅ SFT数据集 UCF_indoor_sft.jsonl 已存在，跳过VL自动标注"
else
    python auto_gen_sft_with_vl.py
fi

echo ""
echo "##################################################################################"
echo "⚠️ 【极轻量人工校验，**不需要播放视频！！**】"
echo "打开文件 UCF_indoor/sft_dataset/UCF_indoor_sft.jsonl"
echo "原理：由Qwen2‑VL多模态模型观看候选切片自动生成描述，存在幻觉风险。"
echo "只做：快速浏览全部行，如果看到明显不合理/幻觉乱描述的条目直接整行删除，不需要修改任何字段内容！"
echo "✅类别、置信度、行为描述全部由脚本+Qwen2‑VL自动生成，不用手动填写任何文字！"
echo "##################################################################################"
read -p "浏览完毕，确认没有明显错误，按回车继续QLoRA微调..."

echo -e "\n===== 步骤6：Qwen2‑VL‑7B QLoRA微调（开题3.3.2） ====="
QLORA_ADAPTER="${PROJECT_ROOT}/UCF_indoor/models/qwen_vl_lora_adapter"
if [ -d "$QLORA_ADAPTER" ];then
    echo "✅ QLoRA微调adapter权重文件夹已存在，是否重新训练？"
    read -p "输入 y 重新训练，其他按键跳过微调：" ans
    if [ "$ans" = "y" ] || [ "$ans" = "Y" ];then
        python qwen_vl_qlora_train.py
    else
        echo "⏭️ 跳过QLoRA微调"
    fi
else
    python qwen_vl_qlora_train.py
fi

echo -e "\n===== 步骤7：端到端系统测试（推理阶段完全无人工） ====="
python end2end_system.py

echo ""
echo "######################## ALL FINISHED! ########################"
echo "✅训练阶段：仅极轻量浏览jsonl，没有逐一看视频写标注；推理阶段输入视频自动输出告警工单，无人工。"
echo "测试其他视频命令：python end2end_system.py 视频完整路径.mp4"
#rm /data/Wcy/code/vad_indoor/UCF_indoor/sft_dataset/UCF_indoor_sft.jsonl /data/Wcy/code/vad_indoor/UCF_indoor/candidate_clips_manifest_filtered.json /data/Wcy/code/vad_indoor/UCF_indoor/candidate_clips_manifest.json /data/Wcy/code/vad_indoor/UCF_indoor/candidate_clips/*.mp4 /data/Wcy/code/vad_indoor/UCF_indoor/models/tcn_best.pth
