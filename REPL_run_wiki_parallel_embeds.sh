#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 <model_name> <source_lang> <target_langs> <experiments>"
    echo ""
    echo "Arguments:"
    echo "  model_name    : Model name (e.g., Alibaba_mGTE)"
    echo "  source_lang   : Source language code (e.g., de)"
    echo "  target_langs  : Comma-separated target language codes (e.g., en,hi,it,ko)"
    echo "  experiments   : Comma-separated experiment numbers (e.g., 1,2,3)"
    echo ""
    echo "Example:"
    echo "  $0 Alibaba_mGTE de en,hi,it,ko 1,2,3"
    exit 1
}

# Check if correct number of arguments provided
if [ $# -ne 4 ]; then
    echo "Error: Incorrect number of arguments"
    usage
fi

# Parse arguments
MODEL_NAME="$1"
SOURCE_LANG="$2"
IFS=',' read -ra TARGET_LANGS <<< "$3"
IFS=',' read -ra EXPERIMENTS <<< "$4"

# Base paths
SCRIPT_PATH="src/fair_sentence_transformers/scripts/compute_embeddings.py"
CONFIG_BASE_PATH="config/${MODEL_NAME}/wiki_parallel"

echo "Running embedding computations for ${MODEL_NAME}"
echo "Source language: ${SOURCE_LANG}"
echo "Target languages: ${TARGET_LANGS[*]}"
echo "Experiments: ${EXPERIMENTS[*]}"
echo "----------------------------------------"

# Loop through experiments and target languages
for exp in "${EXPERIMENTS[@]}"; do
    for target_lang in "${TARGET_LANGS[@]}"; do
        config_file="${CONFIG_BASE_PATH}/embedding_config_wiki_parallel_${exp}_${SOURCE_LANG}_${target_lang}.json"
        
        echo "Running experiment ${exp} for ${SOURCE_LANG} -> ${target_lang}"
        echo "Config: ${config_file}"
        
        poetry run python "${SCRIPT_PATH}" --config "${config_file}"
        
        # Check if command was successful
        if [ $? -eq 0 ]; then
            echo "✓ Successfully completed experiment ${exp} for ${SOURCE_LANG} -> ${target_lang}"
        else
            echo "✗ Failed experiment ${exp} for ${SOURCE_LANG} -> ${target_lang}"
            echo "Do you want to continue? (y/n)"
            read -r response
            if [[ ! "$response" =~ ^[Yy]$ ]]; then
                echo "Stopping execution."
                exit 1
            fi
        fi
        
        echo "----------------------------------------"
    done
done

echo "All embedding computations completed!"