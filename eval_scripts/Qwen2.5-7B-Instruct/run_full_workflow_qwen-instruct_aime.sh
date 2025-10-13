#!/bin/bash
set -e

ROOT="TO_BE_FILLED"

# --- CONFIGURATION ---
MODEL_PATH="TO_BE_FILLED"
MODEL_NAME="Qwen2.5-7B-Instruct"
DATA_ROOT_DIR="${ROOT}/benchmarks/AIME24" 

METHODS_TO_RUN=("self_critique" "dime" "consistency")

# --- Sampling & VLLM Configuration ---
TEMPERATURE_RANDOM=0.8
NUM_RANDOM_SAMPLES=10
TENSOR_PARALLEL_SIZE=4
MAX_NEW_TOKENS=4096
BATCH_SIZE=16

SUBSET_SOURCE="aime"
NUM_SAMPLES_PER_SOURCE=-1

SUBSET_TAG="_${SUBSET_SOURCE:-all}"
SAMPLE_TAG="_n${NUM_SAMPLES_PER_SOURCE}"
if [ "$NUM_SAMPLES_PER_SOURCE" -lt 0 ]; then
    SAMPLE_TAG="_all_samples"
fi
FILENAME_TAG="${SUBSET_TAG}${SAMPLE_TAG}"

# --- Output Configuration ---

RESULTS_DIR="${ROOT}/Final_results/${MODEL_NAME}/${SUBSET_TAG}_${SAMPLE_TAG}"
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

PERTURBATION_PREFIX="The old city district truly came alive after sunset. Cobblestone streets, still warm from the afternoon sun, reflected the soft glow of iron-cast lamps." 
PERTURBATION_SUFFIX="The aroma of freshly baked bread from a corner bakery mingled with the distant sound of a jazz saxophone drifting from an open window."

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

echo "--> Step 2: Evaluating DIME and all baseline methods..."
python evaluate_all_methods.py \
    --input_file "$GENERATED_DATA_FILE" \
    --output_summary_json "$EVAL_SUMMARY_JSON" \
    --output_plot "$PLOT_PNG" \

echo ""
echo "======================================================"
echo "           Workflow Completed!"
echo "======================================================"