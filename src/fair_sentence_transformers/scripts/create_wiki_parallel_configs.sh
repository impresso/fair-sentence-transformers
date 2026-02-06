#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 <output_folder> <model> <experiment_number> <source_langs> <target_langs> [batch_size_standalone] [batch_size_concat]"
    echo ""
    echo "Arguments:"
    echo "  output_folder         : Path where config JSON files will be saved"
    echo "  model                 : Model type (mgte, jina, or qwen3)"
    echo "  experiment_number     : Experiment number (1, 2, 3, 4, ...)"
    echo "  source_langs          : Comma-separated source language codes (e.g., en,de)"
    echo "  target_langs          : Comma-separated target language codes (e.g., en,de,ko)"
    echo "  batch_size_standalone : Optional batch size for standalone (default: 8)"
    echo "  batch_size_concat     : Optional batch size for concat (default: 2)"
    echo ""
    echo "Model configurations:"
    echo "  mgte  : Uses Alibaba-NLP/gte-multilingual-base"
    echo "  jina  : Uses jinaai/jina-embeddings-v3"
    echo "  qwen3 : Uses Qwen/Qwen3-Embedding-0.6B"
    echo ""
    echo "Example:"
    echo "  $0 config/MyModel/wiki_parallel mgte 4 en,de en,de,ko"
    echo "  $0 config/MyModel/wiki_parallel jina 4 en,de en,de,ko 16 4"
    echo "  $0 config/MyModel/wiki_parallel qwen3 4 en,de en,de,ko"
    exit 1
}

# Check minimum number of arguments
if [ $# -lt 5 ]; then
    echo "Error: Insufficient arguments"
    usage
fi

# Parse arguments
OUTPUT_FOLDER="$1"
MODEL="$2"
EXPERIMENT_NUMBER="$3"
IFS=',' read -ra SOURCE_LANGS <<< "$4"
IFS=',' read -ra TARGET_LANGS <<< "$5"
BATCH_SIZE_STANDALONE="${6:-8}"
BATCH_SIZE_CONCAT="${7:-2}"

# Set model-specific configurations
case "$MODEL" in
    "mgte")
        MODEL_NAME="Alibaba-NLP/gte-multilingual-base"
        TOKENIZED_DATASET_PATH="data/wiki_parallel_en_de_hi_it_ko_zh/tokenized__Alibaba-NLP_gte-multilingual-base"
        ;;
    "jina")
        MODEL_NAME="jinaai/jina-embeddings-v3"
        TOKENIZED_DATASET_PATH="data/wiki_parallel_en_de_hi_it_ko_zh/tokenized__jinaai_jina-embeddings-v3"
        ;;
    "qwen3")
        MODEL_NAME="Qwen/Qwen3-Embedding-0.6B"
        TOKENIZED_DATASET_PATH="data/wiki_parallel_en_de_hi_it_ko_zh/tokenized__Qwen_Qwen3-Embedding-0.6B"
        ;;
    *)
        echo "Error: Invalid model '$MODEL'. Must be 'mgte', 'jina', or 'qwen3'"
        exit 1
        ;;
esac

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Combine all unique languages for monolingual configs
ALL_LANGS=($(printf '%s\n' "${SOURCE_LANGS[@]}" "${TARGET_LANGS[@]}" | sort -u))

echo "Creating config files in: $OUTPUT_FOLDER"
echo "Model: $MODEL ($MODEL_NAME)"
echo "Experiment number: $EXPERIMENT_NUMBER"
echo "Source languages: ${SOURCE_LANGS[*]}"
echo "Target languages: ${TARGET_LANGS[*]}"
echo "All languages for monolingual: ${ALL_LANGS[*]}"
echo "Batch sizes: standalone=$BATCH_SIZE_STANDALONE, concat=$BATCH_SIZE_CONCAT"
echo "----------------------------------------"

CONFIG_COUNT=0

# Function to create config file
create_config() {
    local filename="$1"
    local source_lang="$2"
    local target_lang="$3"
    local filepath="$OUTPUT_FOLDER/$filename"
    
    cat > "$filepath" << EOF
{
    "model_name": "$MODEL_NAME",
    "tokenized_dataset_path": "$TOKENIZED_DATASET_PATH",
    "indices_path": "config/wiki_parallel/indices_wiki_parallel_${EXPERIMENT_NUMBER}_en_de_hi_it_ko_zh.json",
    "mode": "wiki_parallel",
    "source_lang": "$source_lang",$(if [ -n "$target_lang" ]; then echo "
    \"target_lang\": \"$target_lang\","; fi)
    "separator": " ",
    "device": "cuda",
    "batch_size_standalone": $BATCH_SIZE_STANDALONE,
    "batch_size_concat": $BATCH_SIZE_CONCAT,
    "embeddings_output_dir": "results/wiki_parallel"
}
EOF
    
    echo "Created: $filename"
    ((CONFIG_COUNT++))
}

# Create monolingual configs (all unique languages)
echo "Creating monolingual configs..."
for lang in "${ALL_LANGS[@]}"; do
    filename="embedding_config_wiki_parallel_${EXPERIMENT_NUMBER}_${lang}.json"
    create_config "$filename" "$lang" ""
done

# Create bilingual configs (source -> target, excluding same language pairs)
echo "Creating bilingual configs..."
for source_lang in "${SOURCE_LANGS[@]}"; do
    for target_lang in "${TARGET_LANGS[@]}"; do
        if [ "$source_lang" != "$target_lang" ]; then
            filename="embedding_config_wiki_parallel_${EXPERIMENT_NUMBER}_${source_lang}_${target_lang}.json"
            create_config "$filename" "$source_lang" "$target_lang"
        fi
    done
done

echo "----------------------------------------"
echo "Successfully created $CONFIG_COUNT config files in $OUTPUT_FOLDER"