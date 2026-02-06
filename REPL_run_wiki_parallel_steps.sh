#!/bin/bash

# Wiki Parallel Processing Pipeline
# This script runs all the steps to process Wikipedia parallel data

set -e  # Exit on any error

echo "Starting Wiki Parallel Processing Pipeline..."

echo "Step 1: Downloading parallel titles..."
sh src/fair_sentence_transformers/utils/parallel_wiki/download_wikipedia_parallel_titles.sh 20250601 de it hi ko zh

# Step 2: Post-process downloaded language titles
echo "Step 2: Processing Wikipedia titles for each language..."

echo "Processing German titles..."
poetry run python src/fair_sentence_transformers/utils/parallel_wiki/process_wikipedia_titles.py data/_wiki_parallel_titles/de_titles.txt

echo "Processing Italian titles..."
poetry run python src/fair_sentence_transformers/utils/parallel_wiki/process_wikipedia_titles.py data/_wiki_parallel_titles/it_titles.txt

echo "Processing Hindi titles..."
poetry run python src/fair_sentence_transformers/utils/parallel_wiki/process_wikipedia_titles.py data/_wiki_parallel_titles/hi_titles.txt

echo "Processing Korean titles..."
poetry run python src/fair_sentence_transformers/utils/parallel_wiki/process_wikipedia_titles.py data/_wiki_parallel_titles/ko_titles.txt

echo "Processing Chinese titles..."
poetry run python src/fair_sentence_transformers/utils/parallel_wiki/process_wikipedia_titles.py data/_wiki_parallel_titles/zh_titles.txt


# Step 3: Create txt file with matching titles
echo "Step 3: Creating parallel titles file..."
poetry run python src/fair_sentence_transformers/utils/parallel_wiki/create_parallel_titles.py data/_wiki_parallel_titles/de_titles_processed.txt data/_wiki_parallel_titles/it_titles_processed.txt data/_wiki_parallel_titles/ko_titles_processed.txt data/_wiki_parallel_titles/hi_titles_processed.txt data/_wiki_parallel_titles/zh_titles_processed.txt --output_dir data/_wiki_parallel_titles/parallel_matches

# Step 4: Create txt file with matching ids
echo "Step 4: Converting titles to page IDs..."
poetry run python src/fair_sentence_transformers/utils/parallel_wiki/title_to_pageid.py data/_wiki_parallel_titles/parallel_matches/parallel_titles_en_de_hi_it_ko_zh.txt

# Step 5: Download articles from Wikipedia dump
echo "Step 5: Downloading parallel articles..."

echo "Downloading articles using page IDs..."
poetry run python src/fair_sentence_transformers/utils/parallel_wiki/download_parallel_articles.py data/_wiki_parallel_titles/parallel_matches/parallel_ids_en_de_hi_it_ko_zh.txt

echo "Downloading articles using title matching..."
poetry run python src/fair_sentence_transformers/utils/parallel_wiki/download_parallel_articles_title_match.py data/_wiki_parallel_titles/parallel_matches/parallel_titles_en_de_hi_it_ko_zh.txt --languages en de hi it ko zh

# Step 6: Unify the datasets
echo "Step 6: Unifying parallel datasets..."
poetry run python src/fair_sentence_transformers/utils/parallel_wiki/unify_parallel_datasets.py --dataset1 data/_wiki_parallel_titles/parallel_matches/parallel_articles_en_de_hi_it_ko_zh.parquet --dataset2 data/_wiki_parallel_titles/parallel_matches/parallel_articles_titleMatch_de_en_hi_it_ko_zh.parquet --parallel_ids data/_wiki_parallel_titles/parallel_matches/parallel_ids_en_de_hi_it_ko_zh.txt

echo "Wiki Parallel Processing Pipeline completed successfully!"