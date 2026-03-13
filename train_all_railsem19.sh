#!/bin/bash

CONFIGS=(
    "configs/mask2former/mask2former_swin-l-in22k-384x384-pre_1xb1-160k_railsem19-1080x1920.py"
    "configs/segformer/segformer_mit-b5_1xb1-160k_railsem19-1080x1920.py"
    "configs/ocrnet/ocrnet_hr48_1xb2-160k_railsem19-1080x1920.py"
    "configs/deeplabv3plus/deeplabv3plus_r101-d8_1xb1-160k_railsem19-1080x1920.py"
    "configs/pspnet/pspnet_r101-d8_1xb1-160k_railsem19-1080x1920.py"
    "configs/upernet/upernet_r101_1xb1-160k_railsem19-1080x1920.py"
    "configs/fcn/fcn_r101-d8_1xb1-160k_railsem19-1080x1920.py"
    "configs/bisenetv2/bisenetv2_fcn_1xb2-160k_railsem19-1080x1920.py"
    "configs/unet/unet-s5-d16_fcn_1xb1-160k_railsem19-1080x1920.py"
)

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    GPUS=(0)
    echo "CUDA_VISIBLE_DEVICES not set, using default: GPU 0"
else
    IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
    echo "Using GPUs: ${GPUS[*]}"
fi

NUM_GPUS=${#GPUS[@]}
echo "Number of GPUs: $NUM_GPUS"
echo "Total models to train: ${#CONFIGS[@]}"
echo ""

mkdir -p logs

train_model() {
    local config="$1"
    local gpu="$2"
    local base_name=$(basename "$config" .py)
    local log_file="logs/${base_name}_output.log"

    echo "[GPU $gpu] Starting: $base_name"
    echo "[GPU $gpu] Log: $log_file"

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

export -f train_model

if [ $NUM_GPUS -eq 1 ]; then
    echo "Single GPU mode: training sequentially"
    echo "=================================================="

    for config in "${CONFIGS[@]}"; do
        train_model "$config" "${GPUS[0]}"
    done
else
    if command -v parallel &> /dev/null; then
        echo "Using GNU parallel for multi-GPU training"
        echo "=================================================="

        > /tmp/train_tasks.txt
        for i in "${!CONFIGS[@]}"; do
            gpu_idx=$((i % NUM_GPUS))
            echo "${CONFIGS[$i]}|${GPUS[$gpu_idx]}" >> /tmp/train_tasks.txt
        done

        cat /tmp/train_tasks.txt | parallel -j "$NUM_GPUS" --colsep '\|' 'train_model {1} {2}'
    else
        echo "GNU parallel not found, using bash background jobs"
        echo "=================================================="

        pids=()
        for i in "${!CONFIGS[@]}"; do
            gpu_idx=$((i % NUM_GPUS))
            gpu="${GPUS[$gpu_idx]}"
            config="${CONFIGS[$i]}"

            train_model "$config" "$gpu" &
            pids+=($!)

            if [ $((i + 1)) -ge $NUM_GPUS ]; then
                wait -n
            fi
        done

        wait
    fi
fi

echo ""
echo "=================================================="
echo "All training jobs processed!"
echo "Logs saved in: logs/"
echo "=================================================="

echo ""
echo "Summary:"
echo "--------"
for config in "${CONFIGS[@]}"; do
    base_name=$(basename "$config" .py)
    log_file="logs/${base_name}_output.log"
    if [ -f "$log_file" ]; then
        if tail -5 "$log_file" | grep -q " successfully\|Checkpoint" 2>/dev/null; then
            echo "[OK]   $base_name"
        else
            echo "[FAIL] $base_name (check logs)"
        fi
    else
        echo "[MISS] $base_name (no log file)"
    fi
done
