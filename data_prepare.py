# prepare_data.py - 修改版 (使用 MsDataset.load)
import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer
import os
import json

# 新增：导入 MsDataset
from modelscope.msdatasets import MsDataset

# --- 配置 ---
LOCAL_MODEL_PATH = "/home/user/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/"

# ModelScope数据集配置
MODELSCOPE_DATASET_NAME = 'krisfu/delicate_medical_r1_data'

# 处理后的数据集保存路径
TOKENIZED_DATASET_OUTPUT_DIR = './qwen3_0_6b_tokenized_dataset'
MAX_SEQ_LENGTH = 1024 # 根据你的数据和显存调整最大序列长度

# --- 加载Tokenizer ---
print(f"Loading tokenizer from local path: {LOCAL_MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# --- 1. 加载原始数据 (使用 ModelScope 数据集) ---
print(f"Loading data from ModelScope: {MODELSCOPE_DATASET_NAME} (subset: default, split: train)...")
try:
    # 使用 MsDataset.load 加载数据
    raw_ms_dataset = MsDataset.load(MODELSCOPE_DATASET_NAME, subset_name='default', split='train')
    # 将 ModelScope 的 Dataset 转换为 Hugging Face 的 Dataset
    dataset = Dataset.from_list(raw_ms_dataset.to_list()) # 或者直接 dataset = Dataset.from_list(raw_ms_dataset)
    print(dataset[0])
    print(f"Loaded {len(dataset)} examples from ModelScope dataset.")
except Exception as e:
    print(f"Error loading data from ModelScope: {e}. Please ensure you have network access and the dataset exists.")    
    exit(0)


# --- 2. 格式化数据以适应Qwen3的对话模板 ---
# 在 prepare_data.py 中，修改 format_data_for_qwen3 函数
def format_data_for_qwen3(examples):
    formatted_texts = []
    # examples 参数是一个字典，其中每个键对应原始数据集的一列（特征），
    # 它的值是该列所有样本的列表。
    # 例如，examples['instruction'] 是所有样本的 instruction 文本列表。

    # 遍历每个样本
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        question = examples["question"][i] # 获取 'question' 字段
        think_content = examples["think"][i] # 获取 'think' 字段
        answer = examples["answer"][i] # 获取 'answer' 字段

        messages = [
            {"role": "system", "content": "你是一个专业的医疗助手。"},
        ]

        # 结合 instruction 和 question 作为用户输入
        user_content = f"{instruction}\n问题：{question}" if question else instruction
        messages.append({"role": "user", "content": user_content})

        # 构建助手的回复，如果包含思考过程
        assistant_content = ""
        if think_content:
            # 如果think_content是思考过程，且你希望模型学习生成它，则包含
            # 注意：<think>标签通常在Qwen3的思考模式下会自动处理，但你也可以手动包含
            assistant_content += f"<think>{think_content}</think>\n"
        assistant_content += answer # 模型的最终答案

        messages.append({"role": "assistant", "content": assistant_content})

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False, # 训练时提供完整的对话历史
            enable_thinking=True # 如果你希望模型学习生成 <think> 标签，设置为 True
        )
        formatted_texts.append(text)
    return {"text": formatted_texts}


print("Formatting data for Qwen3 chat template...")
formatted_dataset = dataset.map(
    format_data_for_qwen3,
    batched=True,
    # 假设转换后的dataset有这些列，如果实际没有或列名不同，需要调整或移除这行
    remove_columns=dataset.column_names
)
print(f"Formatted {len(formatted_dataset)} examples.")

# --- 3. Tokenization ---
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )

print("Tokenizing dataset...")
tokenized_dataset = formatted_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=os.cpu_count(),
    remove_columns=["text"]
)
print(f"Tokenized {len(tokenized_dataset)} examples.")

# --- 4. 保存处理好的数据集 ---
print(f"Saving tokenized dataset to {TOKENIZED_DATASET_OUTPUT_DIR}...")
tokenized_dataset.save_to_disk(TOKENIZED_DATASET_OUTPUT_DIR)
print("Dataset preparation complete.")