import os
import json
import pandas as pd
import argparse
from verl.utils.hdfs_io import makedirs

def make_map_fn(data_source, question_key, answer_key, instruction):
    def process_fn(example, idx):
        question = example[question_key]
        answer = example[answer_key]
        
        # Construct the prompt
        prompt_content = question + " " + instruction
        
        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": prompt_content,
                }
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                "split": "test",
                "index": idx,
                "original_question": question,
                "original_answer": answer
            }
        }
        return data
    return process_fn

def process_dataset(input_files, output_dir, data_source, question_key, answer_key):
    instruction = "Let's think step by step and output the final answer within \\boxed{}."
    
    all_data = []
    global_idx = 0
    
    for input_file in input_files:
        print(f"Processing {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    processed_item = make_map_fn(data_source, question_key, answer_key, instruction)(item, global_idx)
                    all_data.append(processed_item)
                    global_idx += 1
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in {input_file}: {e}")
                    continue

    df = pd.DataFrame(all_data)
    
    makedirs(output_dir)
    output_path = os.path.join(output_dir, 'test.parquet')
    df.to_parquet(output_path)
    print(f"Saved {len(df)} records to {output_path}")

def main():
    # Process AIME 2025
    aime_input_dir = '/root/autodl-tmp/verl/data/AIME2025'
    aime_output_dir = '/root/autodl-tmp/verl/data/aime2025_processed'
    aime_files = [
        os.path.join(aime_input_dir, 'aime2025-I.jsonl'),
        os.path.join(aime_input_dir, 'aime2025-II.jsonl')
    ]
    # Check if files exist
    existing_aime_files = [f for f in aime_files if os.path.exists(f)]
    if existing_aime_files:
        process_dataset(
            input_files=existing_aime_files,
            output_dir=aime_output_dir,
            data_source='aime2025',
            question_key='question',
            answer_key='answer'
        )
    else:
        print(f"No AIME 2025 files found in {aime_input_dir}")

    # Process MATH 500
    math500_input_dir = '/root/autodl-tmp/verl/data/MATH500'
    math500_output_dir = '/root/autodl-tmp/verl/data/math500_processed'
    math500_files = [os.path.join(math500_input_dir, 'test.jsonl')]
    
    existing_math500_files = [f for f in math500_files if os.path.exists(f)]
    if existing_math500_files:
        process_dataset(
            input_files=existing_math500_files,
            output_dir=math500_output_dir,
            data_source='math500',
            question_key='problem',
            answer_key='answer'
        )
    else:
        print(f"No MATH 500 files found in {math500_input_dir}")

if __name__ == "__main__":
    main()
