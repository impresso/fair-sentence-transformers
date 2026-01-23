#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 <config_folder_path>"
    echo ""
    echo "Arguments:"
    echo "  config_folder_path : Path to folder containing config files (e.g., config/Alibaba_mGTE/wiki_parallel)"
    echo ""
    echo "Example:"
    echo "  $0 config/Alibaba_mGTE/wiki_parallel"
    echo "  $0 config/SomeOtherModel/dataset_name"
    exit 1
}

# Check if correct number of arguments provided
if [ $# -ne 1 ]; then
    echo "Error: Incorrect number of arguments"
    usage
fi

# Parse arguments
CONFIG_FOLDER="$1"

# Check if folder exists
if [ ! -d "$CONFIG_FOLDER" ]; then
    echo "Error: Folder '$CONFIG_FOLDER' does not exist"
    exit 1
fi

# Base script path
SCRIPT_PATH="src/locobench/scripts/compute_embeddings.py"

# Find all JSON config files in the folder
CONFIG_FILES=($(find "$CONFIG_FOLDER" -name "*.json" -type f | sort))

# Check if any config files found
if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
    echo "Error: No JSON config files found in '$CONFIG_FOLDER'"
    exit 1
fi

echo "Running embedding computations from folder: $CONFIG_FOLDER"
echo "Found ${#CONFIG_FILES[@]} config files:"
for config in "${CONFIG_FILES[@]}"; do
    echo "  - $(basename "$config")"
done
echo "----------------------------------------"

# Counter for tracking progress
total=${#CONFIG_FILES[@]}
current=1

# Track failed configs
FAILED_CONFIGS=()

# Loop through all config files
for config_file in "${CONFIG_FILES[@]}"; do
    config_name=$(basename "$config_file")
    
    echo "[$current/$total] Running config: $config_name"
    echo "Full path: $config_file"
    
    poetry run python "$SCRIPT_PATH" --config "$config_file"
    
    # Check if command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed: $config_name"
    else
        echo "✗ Failed: $config_name"
        # Record failed config and continue
        FAILED_CONFIGS+=("$config_file")
    fi
    
    echo "----------------------------------------"
    ((current++))
done

echo "All embedding computations completed!"
echo "Processed $total config files from $CONFIG_FOLDER"

# Print summary of failed configs, if any
if [ ${#FAILED_CONFIGS[@]} -gt 0 ]; then
    echo ""
    echo "Summary: ${#FAILED_CONFIGS[@]} failed run(s):"
    for failed in "${FAILED_CONFIGS[@]}"; do
        echo "  - $failed"
    done
else
    echo "All runs succeeded."
fi