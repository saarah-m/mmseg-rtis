#!/bin/bash

# Train all RailSem19 models with support for multiple GPUs via CUDA_VISIBLE_DEVICES
# Usage:
#   Single GPU:  CUDA_VISIBLE_DEVICES=0 ./train_all_railsem19.sh
#   Multi GPU:   CUDA_VISIBLE_DEVICES=0,1,2 ./train_all_railsem19.sh
#   Default:     ./train_all_railsem19.sh (uses GPU 0)

# Array of all 9 RailSem19 configuration files
CONFIGS=(
    "configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb2-80k_railsem19-540x960.py"
    "configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_railsem19-512x1024.py"
    "configs/ocrnet/ocrnet_hr48_4xb2-160k_railsem19-540x960.py"
    "configs/segformer/segformer_mit-b5_8xb1-160k_railsem19-540x960.py"
    "configs/pspnet/pspnet_r101-d8_4xb2-80k_railsem19-540x960.py"
    "configs/bisenetv2/bisenetv2_fcn_1xb20-128k_railsem19-540x960.py"
    "configs/upernet/upernet_r101_4xb2-80k_railsem19-540x960.py"
    "configs/fcn/fcn_r101-d8_4xb2-80k_railsem19-540x960.py"
    "configs/unet/unet-s5-d16_fcn_1xb4-80k_railsem19-540x960.py"
)

# Get available GPUs from CUDA_VISIBLE_DEVICES or default to GPU 0
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    GPUS=(0)
    echo "CUDA_VISIBLE_DEVICES not set, using default: GPU 0"
else
    # Parse comma-separated GPU IDs
    IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
    echo "Using GPUs: ${GPUS[*]}"
fi

NUM_GPUS=${#GPUS[@]}
echo "Number of GPUs: $NUM_GPUS"
echo "Total models to train: ${#CONFIGS[@]}"
echo ""

# Create logs directory
mkdir -p logs

# Function to train a single model on a specific GPU
train_model() {
    local config="$1"
    local gpu="$2"
    local base_name=$(basename "$config" .py)
    local log_file="logs/${base_name}_output.log"

    echo "[GPU $gpu] Starting: $base_name"
    echo "[GPU $gpu] Log: $log_file"

    # Run training on specific GPU
    CUDA_VISIBLE_DEVICES="$gpu" python tools/train.py "$config" > "$log_file" 2>&1
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "[GPU $gpu] Completed successfully: $base_name"
    else
        echo "[GPU $gpu] FAILED (exit $exit_code): $base_name"
        echo "[GPU $gpu] Check log: $log_file"
    fi

    return $exit_code
}

# Export function for use with parallel execution
export -f train_model

# If only 1 GPU, train sequentially
if [ $NUM_GPUS -eq 1 ]; then
    echo "Single GPU mode: training sequentially"
    echo "=================================================="

    for config in "${CONFIGS[@]}"; do
        train_model "$config" "${GPUS[0]}"
    done
else
    # Multiple GPUs: use GNU parallel if available, otherwise simple background jobs
    if command -v parallel &> /dev/null; then
        echo "Using GNU parallel for multi-GPU training"
        echo "=================================================="

        # Create a task list with GPU assignment (round-robin)
        > /tmp/train_tasks.txt
        for i in "${!CONFIGS[@]}"; do
            gpu_idx=$((i % NUM_GPUS))
            echo "${CONFIGS[$i]}|${GPUS[$gpu_idx]}" >> /tmp/train_tasks.txt
        done

        # Run tasks in parallel, max NUM_GPUS concurrent (one per GPU)
        cat /tmp/train_tasks.txt | parallel -j "$NUM_GPUS" --colsep '\|' 'train_model {1} {2}'
    else
        # Fallback: use bash background jobs
        echo "GNU parallel not found, using bash background jobs"
        echo "=================================================="

        # Launch jobs in round-robin fashion
        pids=()
        for i in "${!CONFIGS[@]}"; do
            gpu_idx=$((i % NUM_GPUS))
            gpu="${GPUS[$gpu_idx]}"
            config="${CONFIGS[$i]}"

            train_model "$config" "$gpu" &
            pids+=($!)

            # If we've launched NUM_GPUS jobs, wait for one to finish before continuing
            if [ $((i + 1)) -ge $NUM_GPUS ]; then
                wait -n  # Wait for any job to finish
            fi
        done

        # Wait for all remaining jobs
        wait
    fi
fi

echo ""
echo "=================================================="
echo "All training jobs processed!"
echo "Logs saved in: logs/"
echo "=================================================="

# Print summary of results
echo ""
echo "Summary:"
echo "--------"
for config in "${CONFIGS[@]}"; do
    base_name=$(basename "$config" .py)
    log_file="logs/${base_name}_output.log"
    if [ -f "$log_file" ]; then
        # Check if training completed successfully
        if tail -5 "$log_file" | grep -q " successfully\|Checkpoint" 2>/dev/null; then
            echo "[OK]   $base_name"
        else
            echo "[FAIL] $base_name (check logs)"
        fi
    else
        echo "[MISS] $base_name (no log file)"
    fi
done
