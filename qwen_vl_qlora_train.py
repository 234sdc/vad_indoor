# qwen_vl_qlora_train.py 【适配image路径版本，不再读取视频】
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import torch
from PIL import Image
from datasets import Dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ===================== 配置 =====================
DATA_PATH = "/data/Wcy/code/vad_indoor/UCF_indoor/sft_dataset/UCF_indoor_sft.jsonl"
MODEL_PATH = "/data/Wcy/models/Qwen2-VL-7B-Instruct"
OUTPUT_DIR = "/data/Wcy/code/vad_indoor/UCF_indoor/models/qwen_vl_qlora_vad"

# QLoRA 4bit量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 加载jsonl数据集
def load_vad_jsonl(jsonl_path):
    raw_samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            raw_samples.append({
                "messages": item["messages"]
            })
    return Dataset.from_list(raw_samples)

dataset = load_vad_jsonl(DATA_PATH)

# 加载模型与processor
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# collate_fn：读取图片文件
def collate_fn(examples):
    texts = []
    batch_images = []
    for ex in examples:
        messages = ex["messages"]
        user_content = messages[0]["content"]
        image_list = []
        for seg in user_content:
            if seg["type"] == "image":
                pil_img = Image.open(seg["image"]).convert("RGB")
                image_list.append(pil_img)
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(prompt)
        batch_images.append(image_list)

    batch = processor(
        text=texts,
        images=batch_images,
        return_tensors="pt",
        padding=True
    )
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    return batch

train_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    remove_unused_columns=False
)

trainer = SFTTrainer(
    model=model,
    args=train_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

if __name__ == "__main__":
    print("==== 开始Qwen2-VL QLoRA微调 ====")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"✅训练完成，LoRA权重保存在：{OUTPUT_DIR}")
