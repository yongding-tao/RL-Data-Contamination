"""
Construct MIA (Membership Inference Attack) evaluation set from local GSM8K dataset.
"""

import argparse
import os
import re
import datasets
import pandas as pd
from verl.utils.hdfs_io import copy, makedirs

def extract_solution(solution_str):
    """Extract numeric solution from answer string."""
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    if solution is None:
        return None
    final_solution = solution.group(0).split("#### ")[1].replace(",", "")
    return final_solution

def make_map_fn(is_member):
    """
    Create a processing function that converts raw data to target format and adds 'member' label.
    Args:
        is_member (bool): True means member (contaminated), False means non-member (clean).
    """
    member_label = 1 if is_member else 0
    split_name = "train" if is_member else "test"
    
    def process_fn(example, idx):
        question_raw = example.get("question", "")
        answer_raw = example.get("answer", "")

        instruction_following = 'Let\'s think step by step and output the final answer after "####".'
        question = question_raw + " " + instruction_following
        
        solution = extract_solution(answer_raw)
        if solution is None:
            return None

        data = {
            "data_source": "openai/gsm8k",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "member": member_label,
            "extra_info": {
                "split": split_name,
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
            },
        }
        return data

    return process_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/gsm8k_mia")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()
    
    # --- 1. Load data from local ---
    local_data_path = "your_data_path/gsm8k"
    print(f"Loading GSM8K data from local path '{local_data_path}'...")
    
    data_files = {
        "train": os.path.join(local_data_path, "train.jsonl"),
        "test": os.path.join(local_data_path, "test.jsonl"),
    }
    if not os.path.exists(data_files["train"]) or not os.path.exists(data_files["test"]):
        raise FileNotFoundError(f"Could not find train.jsonl or test.jsonl in '{local_data_path}'.")
        
    dataset = datasets.load_dataset("json", data_files=data_files)
    print("Data loading completed.")

    # --- 2. Simulate data contamination ---
    print("\nStarting data contamination simulation...")
    original_train = dataset["train"]
    original_test = dataset["test"].shuffle(seed=42)

    test_size = len(original_test)
    half_test_size = test_size // 2
    
    test_to_contaminate = original_test.select(range(half_test_size))
    new_test_set = original_test.select(range(half_test_size, test_size))

    # Create contaminated training set (only for generating train.parquet)
    train_contaminated = datasets.concatenate_datasets([original_train, test_to_contaminate])
    
    print(f"Contamination simulation completed:")
    print(f"  - New (contaminated) training set size: {len(train_contaminated)}")
    print(f"  - New (clean) test set size: {len(new_test_set)}")

    # --- 3. Process data and add 'member' labels ---
    print("\nProcessing data for saving...")
    original_processed_test = original_test.map(function=make_map_fn(is_member=False), with_indices=True)
    original_processed_test = original_processed_test.filter(lambda example: example is not None)


    # Process contaminated training set
    processed_train = train_contaminated.map(function=make_map_fn(is_member=True), with_indices=True)
    processed_train = processed_train.filter(lambda example: example is not None)
    
    # Process new clean test set
    processed_test = new_test_set.map(function=make_map_fn(is_member=False), with_indices=True)
    processed_test = processed_test.filter(lambda example: example is not None)

    # --- 4. Create MIA evaluation set (core changes) ---
    print("\nCreating MIA evaluation dataset...")
    # gsm8k_mia_all.parquet: Reconstruct original test set with new member labels
    
    # Label the leaked part with member=1
    processed_contaminated_part = test_to_contaminate.map(function=make_map_fn(is_member=True), with_indices=True)
    processed_contaminated_part = processed_contaminated_part.filter(lambda example: example is not None)

    # Label the clean part with member=0 (this is the same as processed_test)
    processed_clean_part = new_test_set.map(function=make_map_fn(is_member=False), with_indices=True)
    processed_clean_part = processed_clean_part.filter(lambda example: example is not None)

    # Reconcatenate to form complete labeled original test set
    mia_all_dataset = datasets.concatenate_datasets([processed_contaminated_part, processed_clean_part])
    print(f"  - 'gsm8k_mia_all' dataset created, total size: {len(mia_all_dataset)} (should be approximately equal to original test set size)")
    
    # gsm8k_mia_100.parquet: Balanced sampling from mia_all_dataset
    mia_all_members = mia_all_dataset.filter(lambda x: x['member'] == 1)
    mia_all_non_members = mia_all_dataset.filter(lambda x: x['member'] == 0)
    
    members_sample = mia_all_members.shuffle(seed=42).select(range(50))
    non_members_sample = mia_all_non_members.shuffle(seed=42).select(range(50))
    
    mia_100_dataset = datasets.concatenate_datasets([members_sample, non_members_sample]).shuffle(seed=42)
    print(f"  - 'gsm8k_mia_100' dataset created, total size: {len(mia_100_dataset)}")

    # --- 5. Save all files ---
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    print(f"\nSaving all Parquet files to: {local_dir}")

    original_processed_test.to_parquet(os.path.join(local_dir, "original_test.parquet"))

    # Save contaminated train/test split
    processed_train.to_parquet(os.path.join(local_dir, "train.parquet"))
    processed_test.to_parquet(os.path.join(local_dir, "test.parquet"))
    
    # Save MIA evaluation set
    mia_all_dataset.to_parquet(os.path.join(local_dir, "gsm8k_mia_all.parquet"))
    mia_100_dataset.to_parquet(os.path.join(local_dir, "gsm8k_mia_100.parquet"))
    
    print("All files saved.")

    if args.hdfs_dir is not None:
        print(f"Copying local files to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print("Copy completed.")