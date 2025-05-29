# finetune_qwen3.py - 针对 4xRTX 4090 的超参数建议

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling
import os

# --- 1. 配置模型 ---
LOCAL_MODEL_PATH = "/home/user/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/"
TOKENIZED_DATASET_PATH = './qwen3_0_6b_tokenized_dataset'
OUTPUT_DIR = './qwen3_0_6b_medical_full_finetuned'

# --- 训练参数建议 ---
TRAINING_EPOCHS = 5 # 适当增加 epochs，因为数据量可能不大
LEARNING_RATE = 2e-5 # 全量微调的经典学习率
# BATCH_SIZE 和 GRADIENT_ACCUMULATION_STEPS 组合得到有效批次大小
# 假设每张卡上的 batch_size 是 8，梯度累积 4 步，总共 4 张卡
# 有效批次大小 = per_device_train_batch_size * gradient_accumulation_steps * num_gpus
# 8 * 4 * 4 = 128
PER_DEVICE_TRAIN_BATCH_SIZE = 4 # 每张GPU的批次大小，4090 24GB 显存应能支持
GRADIENT_ACCUMULATION_STEPS = 4 # 梯度累积步数
MAX_SEQ_LENGTH = 1024 # 保持不变，或根据实际数据长度调整，不建议过大以免不必要的填充

# --- 2. 加载Tokenizer和模型 ---
print(f"Loading tokenizer from local path: {LOCAL_MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model from local path: {LOCAL_MODEL_PATH}...")
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.bfloat16, # 继续使用 bfloat16 节省显存并利用 Tensor Core
    device_map="auto", # 自动利用所有4张GPU
    trust_remote_code=True
)

# --- 3. 加载预处理好的数据集 ---
print(f"Loading tokenized dataset from {TOKENIZED_DATASET_PATH}...")
tokenized_dataset = load_from_disk(TOKENIZED_DATASET_PATH)
print(f"Loaded {len(tokenized_dataset)} tokenized examples.")

# **重要：划分为训练集和验证集**
# 建议将一小部分数据（如 10%）用作验证集，以便监控训练过程和防止过拟合
if len(tokenized_dataset) > 1000: # 假设数据量足够大才划分
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    print(f"Dataset split: Training {len(train_dataset)} examples, Validation {len(eval_dataset)} examples.")
else:
    print("Dataset too small for explicit train/eval split, using full dataset for training.")
    train_dataset = tokenized_dataset
    eval_dataset = None # 没有验证集

# --- 4. 配置训练参数 ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=TRAINING_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE, # <-- 修改
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, # <-- 修改
    learning_rate=LEARNING_RATE,
    logging_steps=50, # 增加日志频率，方便观察
    save_steps=500, # 可以根据训练数据量调整保存步数，如果数据小，可以设为总步数/2
    # save_strategy="epoch", # 对于小数据集，按epoch保存可能更合理
    fp16=False,
    bf16=True,
    report_to="tensorboard", # 建议设置为 "tensorboard" 进行可视化监控
    save_total_limit=3, # 保留更多的检查点
    eval_strategy="steps" if eval_dataset else "no", # 如果有验证集，开启评估
    eval_steps=500 if eval_dataset else None, # 评估步数
    load_best_model_at_end=True if eval_dataset else False, # 如果有验证集，在训练结束时加载最佳模型
    metric_for_best_model="loss", # 监控验证集损失
    greater_is_better=False, # 损失越小越好
    # multi_gpu = True 隐式开启，因为 device_map="auto" 和 accelerate
    # 其他可能对多卡有益的参数：
    # dataloader_num_workers=os.cpu_count() // 2, # 数据加载工作进程数，避免CPU成为瓶颈
    # sharded_ddp="simple", # 对于多卡训练，可以考虑sharded_ddp来进一步节省显存（尤其适用于大模型）
    # half_precision_backend="auto", # 默认即可
)

# --- 5. 创建Trainer并开始训练 ---
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, # 传入验证集
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

# --- 6. 保存微调后的模型 ---
trainer.save_model(OUTPUT_DIR)
print(f"Fine-tuned model saved to {OUTPUT_DIR}")

# --- 7. 加载并测试模型 (可选) ---
print("\n--- Loading fine-tuned model for testing ---")
# 全量微调保存的是整个模型，所以直接用 AutoModelForCausalLM 加载
finetuned_model = AutoModelForCausalLM.from_pretrained( # <-- 这里变回 AutoModelForCausalLM
    OUTPUT_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
finetuned_model = finetuned_model.eval()

def generate_medical_response(prompt_text, max_new_tokens=256):
    messages = [
        {"role": "system", "content": "你是一个专业的医疗助手。"},
        {"role": "user", "content": prompt_text}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # 如果你希望模型在推理时也使用思考模式，且训练数据中包含思考过程
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(finetuned_model.device)

    generated_ids = finetuned_model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.8,
        temperature=0.7,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return generated_text.strip()

# 测试
print("\n--- Model Testing ---")
print("Q: 糖尿病的早期症状有哪些？")
print("A:", generate_medical_response("糖尿病的早期症状有哪些？"))

print("\nQ: 心脏病发作时该怎么做？")
print("A:", generate_medical_response("心脏病发作时该怎么做？"))