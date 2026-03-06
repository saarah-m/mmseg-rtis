#!/bin/bash

# Array of all 9 RailSem19 configuration files
CONFIGS=(
    "configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb2-80k_railsem19-512x1024.py"
    "configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_railsem19-512x1024.py"
    "configs/ocrnet/ocrnet_hr48_4xb2-160k_railsem19-512x1024.py"
    "configs/segformer/segformer_mit-b5_8xb1-160k_railsem19-1024x1024.py"
    "configs/pspnet/pspnet_r101-d8_4xb2-80k_railsem19-512x1024.py"
    "configs/bisenetv2/bisenetv2_fcn_4xb8-320k_railsem19-1024x1024.py"
    "configs/upernet/upernet_r101_4xb2-80k_railsem19-512x1024.py"
    "configs/fcn/fcn_r101-d8_4xb2-80k_railsem19-512x1024.py"
    "configs/unet/unet-s5-d16_fcn_4xb4-160k_railsem19-512x1024.py"
)

echo "Starting sequential training of all 9 RailSem19 models..."

for config in "${CONFIGS[@]}"; do
    echo "=================================================================================="
    echo "Training model using config: $config"
    echo "=================================================================================="
    
    # Setup log file name based on config
    base_name=$(basename "$config" .py)
    log_file="${base_name}_output.log"
    echo "Writing output to: $log_file"

    # Run the training script. We allow it to continue to the next model even if one fails.
    python tools/train.py "$config" > "$log_file" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "Successfully completed training for $config"
    else
        echo "WARNING: Training failed for $config. Continuing to next model..."
    fi
done

echo "All training jobs processed!"
