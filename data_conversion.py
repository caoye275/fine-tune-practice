# 将r1_data_example.jsonl转换为sft_r1_data_example.jsonl
import json
import os

def convert_to_alpaca_format(input_filepath, output_filepath):
    """
    Converts a custom JSONL dataset with 'instruction', 'question', 'think', 'answer'
    fields to Alpaca-style JSONL format with 'instruction', 'input', 'output' fields.

    Args:
        input_filepath (str): Path to your original .jsonl dataset file.
        output_filepath (str): Path where the new Alpaca-formatted .jsonl file will be saved.
    """
    converted_data = []

    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    original_record = json.loads(line.strip())

                    # Extract fields, providing default empty strings if a field is missing
                    instruction = original_record.get('instruction', '')
                    question = original_record.get('question', '')
                    answer = original_record.get('answer', '')

                    # Map to Alpaca format
                    alpaca_record = {
                        "instruction": instruction,
                        "input": question,  # Map 'question' to 'input'
                        "output": answer    # 'answer' already contains 'think' and 'response'
                    }
                    converted_data.append(alpaca_record)
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed JSON line: {line.strip()} - Error: {e}")
                except KeyError as e:
                    print(f"Skipping line due to missing expected key: {e} in {line.strip()}")

        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            for record in converted_data:
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"Successfully converted {len(converted_data)} records.")
        print(f"Alpaca-formatted data saved to: {output_filepath}")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_filepath}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Configuration ---
# Your original data file (e.g., my_cot_data.jsonl)
input_file = 'r1_data_example.jsonl'
# The desired output file name for Alpaca format
output_file = 'sft_r1_data_example.jsonl'

# Create a dummy input file for demonstration if it doesn't exist
if not os.path.exists(input_file):
    print(f"Creating a dummy input file: {input_file}")
    dummy_data = [
        {"instruction": "请总结以下文本的主要观点：", "question": "文本：人工智能（AI）正在迅速发展，并在各个领域产生深远影响。", "think": "用户要求总结文本，我需要提取核心信息。", "answer": "思考：我需要从文本中提取主要思想，重点关注核心信息和支持细节。我将逐节阅读并提炼关键信息。\n\n回答：该文本的主要观点是人工智能发展迅速，并对各个领域产生深远影响。"},
        {"instruction": "计算 123 和 456 的和。", "question": "", "think": "这是一个简单的加法。", "answer": "思考：这是一个简单的加法问题。我将 123 和 456 相加。\n\n回答：123 和 456 的和是 579。"},
        {"instruction": "解释量子纠缠的概念。", "question": "量子力学概念", "think": "需要清晰定义并举例。", "answer": "思考：我需要定义量子纠缠，并以易于理解的方式解释其关键特性，包括幽灵般的超距作用及其含义。\n\n回答：量子纠缠是一种现象，其中两个或多个粒子以某种方式相互连接，无论它们之间的距离如何，它们都共享相同的命运。当你测量一个纠缠粒子的状态时，你立即就知道另一个粒子的状态，即使它们相距数光年。"}
    ]
    with open(input_file, 'w', encoding='utf-8') as f:
        for item in dummy_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("Dummy file created. Running conversion...")

# --- Run the conversion ---
convert_to_alpaca_format(input_file, output_file)

# You can optionally print the first few lines of the output file to verify
print("\n--- First 3 lines of the converted Alpaca file: ---")
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(line.strip())