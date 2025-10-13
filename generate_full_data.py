import json
import argparse
import os
import glob
import pandas as pd
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import copy

def get_user_content(prompt_obj):
    if isinstance(prompt_obj, np.ndarray): prompt_obj = prompt_obj.tolist()
    if isinstance(prompt_obj, list) and len(prompt_obj) > 0 and 'content' in prompt_obj[-1]:
        return prompt_obj[-1]['content']
    return None

def format_prompt(message, template, tokenizer):
    if not message: return ""
    question = message[-1]['content']
    if template == 'prime_sft':
        content = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
        msg = [{"role": "user", "content": content}]
        if len(message) > 1 and message[0]['role'] == 'system':
            msg.insert(0, message[0])
        return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    elif template == 'own':
        return tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    else:
        raise ValueError(f'Unknown template: {template}')

def process_single_output(output, tokenizer, approx_mode: str = "renorm"):
    """
    Process a single VLLM completion object, calculate and return all required metrics in real-time.
    No longer save the massive full_logprobs_dist.
    """
    all_step_logprobs = output.logprobs if output.logprobs is not None else []
    
    # --- 1. Extract logprobs of actually generated tokens ---
    actual_logprobs = []
    for i, step_logprobs_dict in enumerate(all_step_logprobs):
        token_id = output.token_ids[i]
        if token_id in step_logprobs_dict:
            actual_logprobs.append(step_logprobs_dict[token_id].logprob)
        else:
            actual_logprobs.append(None) # If not in Top-K, mark as None
    
    # --- 2. Calculate Entropy, Mu, and Sigma for each token in real-time ---
    entropies, mus, sigmas = [], [], []
    for step_dist in all_step_logprobs:
        # If a step has no logprobs (e.g., encountering EOS), skip it
        if not step_dist:
            continue
        
        # Extract logprobs from Top-K distribution and calculate probs
        step_logprobs = np.array([p.logprob for p in step_dist.values()])
        step_probs = np.exp(step_logprobs)
        
        # Calculate Entropy
        step_entropy = -np.sum(step_probs * np.log(step_probs))
        entropies.append(step_entropy)
        
        # Calculate Mu (E[log P])
        mu = np.sum(step_probs * step_logprobs)
        mus.append(mu)
        
        # Calculate Sigma (sqrt(E[(log P)^2] - (E[log P])^2))
        sigma_sq = np.sum(step_probs * np.square(step_logprobs)) - np.square(mu)
        sigma = np.sqrt(max(sigma_sq, 1e-6)) # Avoid negative square root
        sigmas.append(sigma)

    return {
        "generated_text": output.text,
        "logprobs": actual_logprobs, # for ppl and Min-K%
        "mus": mus, # for Min-K%++
        "sigmas": sigmas, # for Min-K%++
        "token_ids": output.token_ids, # for CDD
        "entropies": entropies, # for DIME
    }

def main():
    parser = argparse.ArgumentParser(description="Generate three types of responses for each prompt: Original Greedy, Perturbed Greedy, Original Random.")
    # --- All required parameters ---
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_root_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--prompt_template", type=str, default="own")
    parser.add_argument("--perturbation_prefix", type=str, default="[SYSTEM NOTE: Please double check your reasoning.]")
    parser.add_argument("--perturbation_suffix", type=str, default="[END OF QUERY]")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature_random", type=float, default=0.8)
    parser.add_argument("--num_random_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--subset_source", type=str, default=None)
    parser.add_argument("--num_samples_per_source", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--K", type=int, default=50)
    parser.add_argument("--methods_to_run", type=str, nargs='+', required=True, 
                        choices=['dime', 'consistency', 'self_critique', 'self_critique_ablation'])
    parser.add_argument("--approx_mode", type=str, default="renorm", choices=["renorm", "rest"])
    args = parser.parse_args()
    
    # --- 1. Resume from checkpoint ---
    processed_contents = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            for line in f:
                try: processed_contents.add(json.loads(line)['original_user_content'])
                except: pass
        print(f"Found {len(processed_contents)} already processed prompts, will skip automatically.")

    # --- 2. Load, filter and sample ---
    print(f"Searching for data files in {args.data_root_dir}...")
    all_tasks = [row.to_dict() for ext in ['*.jsonl', '*.parquet'] for f in glob.glob(os.path.join(args.data_root_dir, '**', ext), recursive=True) for _, row in (pd.read_parquet(f) if f.endswith('.parquet') else pd.read_json(f, lines=True)).iterrows() if 'prompt' in row and 'member' in row]
    df_all = pd.DataFrame(all_tasks)

    print('df_all.keys()', df_all.keys())

    if args.subset_source:
        df_filtered = df_all[df_all['data_source'] == args.subset_source].copy()
    else:
        df_filtered = df_all

    if args.num_samples_per_source > 0:
        df_sampled = df_filtered.groupby('data_source').apply(lambda x: x.sample(n=min(len(x), args.num_samples_per_source), random_state=42)).reset_index(drop=True)
    else:
        df_sampled = df_filtered

    tasks_to_process = [row for _, row in df_sampled.iterrows() if get_user_content(row['prompt']) and get_user_content(row['prompt']) not in processed_contents]

    if not tasks_to_process:
        print("All target prompts have been processed or not found. Program exiting.")
        return
    print(f"\nNumber of new prompts to process: {len(tasks_to_process)}")
    print(f"Data source distribution to process:\n{pd.Series([t['data_source'] for t in tasks_to_process]).value_counts()}")
    
    # --- 3. Load VLLM ---
    logprobs_to_request = args.K
    print(f"Loading model: {args.model_path} (TP={args.tensor_parallel_size})...")
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True, gpu_memory_utilization=0.9, max_logprobs=logprobs_to_request, dtype='bfloat16')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # Get model's maximum length limit
    max_model_len = llm.llm_engine.model_config.max_model_len
    print(f"Detected model maximum length: {max_model_len}")
    
    # --- 4. Define sampling strategies ---
    greedy_params = SamplingParams(temperature=0, n=1, max_tokens=args.max_tokens, logprobs=logprobs_to_request)
    random_params = SamplingParams(temperature=args.temperature_random, n=args.num_random_samples, max_tokens=args.max_tokens, logprobs=logprobs_to_request, top_p=0.95)
    
    # --- 5. Incremental processing and saving ---
    with open(args.output_file, 'a') as f_out:
        for i in tqdm(range(0, len(tasks_to_process), args.batch_size), desc="Processing Batches"):
            batch_tasks = tasks_to_process[i:i+args.batch_size]
            
            # Prepare two sets of prompts: original and perturbed
            batch_original_prompts_formatted = []
            batch_perturbed_prompts_formatted = []
            for task in batch_tasks:
                original_prompt = task['prompt'].tolist() if isinstance(task['prompt'], np.ndarray) else task['prompt']
                perturbed_prompt = copy.deepcopy(original_prompt)
                user_content = get_user_content(original_prompt)
                perturbed_content = f"{args.perturbation_prefix} {user_content} {args.perturbation_suffix}".strip()
                perturbed_prompt[-1]['content'] = perturbed_content
                
                batch_original_prompts_formatted.append(format_prompt(original_prompt, args.prompt_template, tokenizer))
                batch_perturbed_prompts_formatted.append(format_prompt(perturbed_prompt, args.prompt_template, tokenizer))

            # 1. Original Greedy sampling (required by all methods)
            try:
                original_greedy_outputs = llm.generate(batch_original_prompts_formatted, greedy_params)
            except Exception as e:
                print(f"Batch {i // args.batch_size} original Greedy sampling failed: {e}")
                continue

            # 2. Perturbed Greedy sampling
            perturbed_greedy_outputs = None
            if 'dime' in args.methods_to_run:
                try:
                    perturbed_greedy_outputs = llm.generate(batch_perturbed_prompts_formatted, greedy_params)
                except Exception as e:
                    print(f"Batch {i // args.batch_size} perturbed Greedy sampling failed: {e}")
                    perturbed_greedy_outputs = None # Ensure None after failure

            # 3. Original Random sampling (only when consistency is needed)
            original_random_outputs = None
            if 'consistency' in args.methods_to_run:
                try:
                    original_random_outputs = llm.generate(batch_original_prompts_formatted, random_params)
                except Exception as e:
                    print(f"Batch {i // args.batch_size} original Random sampling failed: {e}")
                    original_random_outputs = None

            # 4. Self-critique Greedy sampling (only when self_critique is needed)
            critique_greedy_outputs = None
            if 'self_critique' in args.methods_to_run:
                batch_critique_prompts = []
                SELF_CRITIQUE_INSTRUCTION = "\nA possible answer is provided below (it may or may not be correct). Please provide a response that follows a different reasoning path or provides an alternative solution:\n---\n{response}\n---\nPlease now provide your new, different response:"
                for j in range(len(batch_tasks)):
                    first_pass_text = original_greedy_outputs[j].outputs[0].text
                    task = batch_tasks[j]
                    original_prompt = task['prompt'].tolist() if isinstance(task['prompt'], np.ndarray) else task['prompt']
                    critique_prompt = copy.deepcopy(original_prompt)
                    template_prompt_formatted = format_prompt(critique_prompt, args.prompt_template, tokenizer)
                    template_token_ids = tokenizer.encode(template_prompt_formatted)
                    max_response_len = max_model_len - len(template_token_ids) - 50

                    # As the model context window is limited, we may need to truncate the response
                    response_token_ids = tokenizer.encode(first_pass_text)
                    if len(response_token_ids) > max_response_len:
                        truncated_response_ids = response_token_ids[:max_response_len]
                        truncated_response_text = tokenizer.decode(truncated_response_ids)
                        print('Warning: truncated response due to context window limit')
                    else:
                        truncated_response_text = first_pass_text


                    user_content = get_user_content(original_prompt)
                    new_user_content = user_content + SELF_CRITIQUE_INSTRUCTION.format(response=truncated_response_text)
                    critique_prompt[-1]['content'] = new_user_content
                    batch_critique_prompts.append(format_prompt(critique_prompt, args.prompt_template, tokenizer))
                try:
                    greedy_critique_params = SamplingParams(temperature=args.temperature, n=1, max_tokens=args.max_tokens*2, logprobs=logprobs_to_request)
                    critique_greedy_outputs = llm.generate(batch_critique_prompts, greedy_critique_params)
                except Exception as e:
                    print(f"Batch {i // args.batch_size} self-critique sampling failed: {e}")
                    critique_greedy_outputs = None

            # 5. Ablation version "unfamiliar/unconventional method" Greedy sampling (without concatenating first-pass answer content)
            unfamiliar_greedy_outputs = None
            if 'self_critique_ablation' in args.methods_to_run:
                batch_unfamiliar_prompts = []
                UNFAMILIAR = "Answer using a technique you’d typically avoid or a deliberately unconventional line of reasoning."
                for task in batch_tasks:
                    original_prompt = task['prompt'].tolist() if isinstance(task['prompt'], np.ndarray) else task['prompt']
                    prompt2 = copy.deepcopy(original_prompt)
                    user_content = get_user_content(original_prompt)
                    # Only append instruction, without first-pass answer
                    new_user = f"{user_content}\n\n{UNFAMILIAR}"
                    prompt2[-1]['content'] = new_user
                    batch_unfamiliar_prompts.append(format_prompt(prompt2, args.prompt_template, tokenizer))
                try:
                    unfamiliar_params = SamplingParams(
                        temperature=args.temperature, n=1,
                        max_tokens=args.max_tokens*2, logprobs=logprobs_to_request
                    )
                    unfamiliar_greedy_outputs = llm.generate(batch_unfamiliar_prompts, unfamiliar_params)
                except Exception as e:
                    print(f"Batch {i // args.batch_size} unfamiliar method ablation sampling failed: {e}")
                    unfamiliar_greedy_outputs = None

            for j in range(len(batch_tasks)):
                task = batch_tasks[j]
                final_item = {
                    "original_user_content": get_user_content(task['prompt']),
                    "ground_truth_label": 1 if task['member'] else 0,
                    "data_source": task['data_source'],
                }
                
                # Add result fields as needed
                final_item["original_greedy_results"] = [process_single_output(o, tokenizer, approx_mode=args.approx_mode) for o in original_greedy_outputs[j].outputs]
                
                if perturbed_greedy_outputs:
                    final_item["perturbed_greedy_results"] = [process_single_output(o, tokenizer, approx_mode=args.approx_mode) for o in perturbed_greedy_outputs[j].outputs]
                
                if original_random_outputs:
                    final_item["original_random_results"] = [process_single_output(o, tokenizer, approx_mode=args.approx_mode) for o in original_random_outputs[j].outputs]

                if critique_greedy_outputs:
                    final_item["critique_greedy_results"] = [process_single_output(o, tokenizer, approx_mode=args.approx_mode) for o in critique_greedy_outputs[j].outputs]

                if unfamiliar_greedy_outputs:
                    final_item["unfamiliar_greedy_results"] = [process_single_output(o, tokenizer, approx_mode=args.approx_mode) for o in unfamiliar_greedy_outputs[j].outputs]

                f_out.write(json.dumps(final_item) + '\n')
                f_out.flush()
                os.fsync(f_out.fileno())

    print(f"\nAll data successfully saved to: {args.output_file}")

if __name__ == "__main__":
    main()