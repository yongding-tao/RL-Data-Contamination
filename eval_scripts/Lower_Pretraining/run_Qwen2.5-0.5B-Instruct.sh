#!/bin/bash
set -e

ROOT="TO_BE_FILLED"

export CUDA_VISIBLE_DEVICES=6,7

# --- CONFIGURATION ---
MODEL_PATH="TO_BE_FILLED"
MODEL_NAME="Qwen2.5-0.5B-Instruct-gsm8k"
DATA_ROOT_DIR="${ROOT}/GSM8k"

METHODS_TO_RUN=("self_critique" "dime" "consistency")

# --- Perturbation Configuration ---
PERTURBATION_PREFIX="hello, what's your name?" 
PERTURBATION_SUFFIX="I'm fine, thank you."

# --- Sampling & VLLM Configuration ---
TEMPERATURE_RANDOM=0.8
NUM_RANDOM_SAMPLES=10
TENSOR_PARALLEL_SIZE=2
MAX_NEW_TOKENS=4096
BATCH_SIZE=16

SUBSET_SOURCE="openai/gsm8k"
NUM_SAMPLES_PER_SOURCE=-1

SUBSET_TAG="_${SUBSET_SOURCE:-all}"
SAMPLE_TAG="_n${NUM_SAMPLES_PER_SOURCE}"
if [ "$NUM_SAMPLES_PER_SOURCE" -lt 0 ]; then
    SAMPLE_TAG="_all_samples"
fi
FILENAME_TAG="${SUBSET_TAG}${SAMPLE_TAG}"

# --- Output Configuration ---
RESULTS_DIR="${ROOT}/gsm8k_results/${MODEL_NAME}/${SUBSET_TAG}_${SAMPLE_TAG}"
mkdir -p "$RESULTS_DIR"

GENERATED_DATA_FILE="${RESULTS_DIR}/generated_data.jsonl"
EVAL_SUMMARY_JSON="${RESULTS_DIR}/evaluation_summary.json"
PLOT_PNG="${RESULTS_DIR}/performance_plot.png"
DIME_DETAIL_JSONL="${RESULTS_DIR}/dime_detail_report.jsonl"

# --- WORKFLOW ---
echo "======================================================"
echo "    Starting Final Contamination Detection Workflow"
echo "    Config -> Subset: ${SUBSET_SOURCE:-all}, Samples: ${NUM_SAMPLES_PER_SOURCE}"
echo "======================================================"

CMD_ARGS=""
if [ -n "$SUBSET_SOURCE" ]; then
    CMD_ARGS="$CMD_ARGS --subset_source $SUBSET_SOURCE"
fi
if [ "$NUM_SAMPLES_PER_SOURCE" -ge 0 ]; then
    CMD_ARGS="$CMD_ARGS --num_samples_per_source $NUM_SAMPLES_PER_SOURCE"
fi

echo "--> Step 1: Generating all necessary data..."
python generate_full_data.py \
    --model_path "$MODEL_PATH" \
    --data_root_dir "$DATA_ROOT_DIR" \
    --output_file "$GENERATED_DATA_FILE" \
    --perturbation_prefix "$PERTURBATION_PREFIX" \
    --perturbation_suffix "$PERTURBATION_SUFFIX" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --max_tokens "$MAX_NEW_TOKENS" \
    --temperature_random "$TEMPERATURE_RANDOM" \
    --num_random_samples "$NUM_RANDOM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --methods_to_run "${METHODS_TO_RUN[@]}" \
    $CMD_ARGS

echo "--> Step 1 finished. All data saved to '$GENERATED_DATA_FILE'."
echo ""

subset_by_detector="ppl"         
subset_take="low"                

echo "--> Step 2: Evaluating all methods across quantiles (1.0 → 0.1) using ${subset_by_detector} (${subset_take})..."


EVAL_SUMMARY_JSON="${RESULTS_DIR}/full_eval.json"
PLOT_PNG="${RESULTS_DIR}/full_roc.png"


subset_by_detector="ppl"
subset_take="low"

echo "Subset by detector: ${subset_by_detector}, take: ${subset_take}"

for q in $(seq 1.0 -0.1 0.1); do
  q_tag="${q/./p}"
  echo ">>> Running at quantile=${q}  (tag: q${q_tag}, ${subset_take})"

  python evaluate_all_methods_lower_pretraining.py \
    --input_file "$GENERATED_DATA_FILE" \
    --output_summary_json "$EVAL_SUMMARY_JSON" \
    --output_plot "$PLOT_PNG" \
    --output_self_critique_y_scores "${RESULTS_DIR}/self_critique_y_scores.jsonl" \
    --output_summary_json_subset "${RESULTS_DIR}/subset_eval_${subset_by_detector}_q${q_tag}_${subset_take}.json" \
    --output_plot_subset "${RESULTS_DIR}/subset_roc_${subset_by_detector}_q${q_tag}_${subset_take}.png" \
    --subset_by_detector "${subset_by_detector}" \
    --subset_quantile "${q}" \
    --subset_take "${subset_take}" \
    --with_random_control \
    --random_seed 2025 \
    --output_summary_json_random "${RESULTS_DIR}/subset_eval_${subset_by_detector}_q${q_tag}_${subset_take}_RANDOM.json" \
    --output_plot_random "${RESULTS_DIR}/subset_roc_${subset_by_detector}_q${q_tag}_${subset_take}_RANDOM.png"
done


echo ""
echo "======================================================"
echo "           Workflow Completed!"
echo "======================================================"