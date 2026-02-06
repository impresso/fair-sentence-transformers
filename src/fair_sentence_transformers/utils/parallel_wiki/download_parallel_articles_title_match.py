#!/usr/bin/env python3
"""
Script to download parallel Wikipedia articles based on parallel titles file.

This script takes a .txt file with parallel Wikipedia titles and downloads the 
corresponding articles from Hugging Face Wikipedia datasets in streaming mode.
It creates a unified dataset with specified languages and additional metadata columns.

Usage:
    python download_parallel_articles_title_match.py <input_titles_file> --languages <lang1> <lang2> ...
    
Example:
    python download_parallel_articles_title_match.py /path/to/parallel_titles_en_de_hi_it_ko.txt --languages en de
    python download_parallel_articles_title_match.py /path/to/parallel_titles_en_de_hi_it_ko.txt --languages en
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import sys

from datasets import Dataset, load_dataset
from tqdm import tqdm


def parse_titles_file(file_path: str, target_languages: List[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Parse the parallel titles file and extract language codes and titles.
    
    Args:
        file_path: Path to the parallel titles file
        target_languages: List of language codes to process
        
    Returns:
        Tuple of (language_codes, titles_by_language)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Parse header to get language codes
    header = lines[0].strip()
    all_languages = [lang.strip() for lang in header.split("|||")]
    
    # Validate that all target languages exist in the file
    for lang in target_languages:
        if lang not in all_languages:
            raise ValueError(f"Language '{lang}' not found in file. Available languages: {all_languages}")
    
    # Get indices for target languages
    lang_indices = {lang: all_languages.index(lang) for lang in target_languages}
    
    # Parse titles for target languages only
    titles_by_language = {lang: [] for lang in target_languages}
    valid_article_indices = []
    
    for line_idx, line in enumerate(lines[1:]):
        line = line.strip()
        if not line:
            continue
            
        titles = [title.strip() for title in line.split("|||")]
        if len(titles) != len(all_languages):
            print(f"Warning: Skipping malformed line {line_idx + 2}: {line}")
            continue
            
        # Extract titles for target languages and check if all are present (non-empty)
        target_titles = {lang: titles[lang_indices[lang]] for lang in target_languages}
        
        if all(title for title in target_titles.values()):
            for lang, title in target_titles.items():
                titles_by_language[lang].append(title)
            valid_article_indices.append(line_idx)
        else:
            print(f"Warning: Skipping line {line_idx + 2} with missing titles: {line}")
    
    print(f"Loaded {len(valid_article_indices)} valid parallel articles for languages: {target_languages}")
    
    return target_languages, titles_by_language


def download_articles_for_language(language: str, titles: List[str]) -> List[Dict]:
    """
    Download Wikipedia articles for a specific language using streaming.
    
    Args:
        language: Language code (e.g., 'en', 'de', 'hi', 'it', 'ko')
        titles: List of titles to collect
        
    Returns:
        List of article dictionaries
    """
    print(f"Loading Wikipedia dataset for language: {language}")
    
    # Load dataset in streaming mode
    dataset = load_dataset(
        "wikimedia/wikipedia", 
        f"20231101.{language}", 
        streaming=True,
        split="train"
    )
    
    articles = []
    titles_set = set(titles)
    found_titles = set()
    
    print(f"Processing {len(titles)} titles for {language}")
    print("Streaming through entire dataset to find matching titles...")
    
    with tqdm(desc=f"Processing {language}", unit="articles") as pbar:
        for article in dataset:
            article_title = article["title"]
            
            # Check if this article title matches any of our target titles
            if article_title in titles_set:
                articles.append({
                    "language": language,
                    "id": article["id"],
                    "url": article["url"],
                    "title": article["title"],
                    "text": article["text"]
                })
                found_titles.add(article_title)
                pbar.update(1)
                
                # If we've found all titles, we can break early
                if len(found_titles) == len(titles_set):
                    break
            
            # Update progress bar periodically (every 10000 articles)
            if pbar.n % 10000 == 0:
                pbar.set_postfix({
                    "found": len(found_titles),
                    "target": len(titles_set)
                })
    
    print(f"Found {len(articles)} articles for {language}")
    
    # Report missing titles
    missing_titles = titles_set - found_titles
    if missing_titles:
        print(f"Warning: {len(missing_titles)} titles not found for {language}:")
        for title in sorted(missing_titles):
            print(f"  - {title}")
    
    return articles


def create_unified_dataset(languages: List[str], titles_by_language: Dict[str, List[str]]) -> Dataset:
    """
    Create a unified dataset with all languages and additional metadata columns.
    
    Args:
        languages: List of language codes
        titles_by_language: Dictionary mapping language codes to titles
        
    Returns:
        Unified Dataset object
    """
    all_articles = []
    
    # Download articles for each language
    articles_by_language = {}
    for language in languages:
        articles_by_language[language] = download_articles_for_language(
            language, titles_by_language[language]
        )
    
    # Create mapping from title to article_id (parallel article group index)
    title_to_article_id = {}
    for article_id in range(len(titles_by_language[languages[0]])):
        for lang in languages:
            title = titles_by_language[lang][article_id]
            title_to_article_id[f"{lang}_{title}"] = article_id
    
    # Combine all articles with metadata
    idx = 0
    for language in languages:
        for article in articles_by_language[language]:
            # Find the article_id for this article
            key = f"{language}_{article['title']}"
            article_id = title_to_article_id.get(key, -1)
            
            article_with_metadata = {
                "idx": idx,
                "article_id": article_id,
                "language": article["language"],
                "id": article["id"],
                "url": article["url"],
                "title": article["title"],
                "text": article["text"]
            }
            all_articles.append(article_with_metadata)
            idx += 1
    
    # Create dataset
    return Dataset.from_list(all_articles)


def main():
    parser = argparse.ArgumentParser(
        description="Download parallel Wikipedia articles from titles file"
    )
    parser.add_argument(
        "input_file",
        help="Path to the parallel titles .txt file"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        required=True,
        help="Language codes to process (e.g., 'en de' or just 'en')"
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory (default: same directory as input file)",
        default=None
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse the titles file
    print(f"Parsing titles file: {input_path}")
    languages, titles_by_language = parse_titles_file(str(input_path), args.languages)
    
    print(f"Processing languages: {languages}")
    print(f"Number of parallel articles: {len(titles_by_language[languages[0]])}")
    
    # Create unified dataset
    print("Creating unified dataset...")
    dataset = create_unified_dataset(languages, titles_by_language)
    
    # Save dataset
    lang_code_str = "_".join(sorted(languages))
    output_name = f"parallel_articles_titleMatch_{lang_code_str}"
    output_path = output_dir / f"{output_name}.parquet"
    
    print(f"Saving dataset to: {output_path}")
    dataset.to_parquet(str(output_path))
    
    print(f"Dataset created successfully!")
    print(f"Total samples: {len(dataset)}")
    print(f"Languages: {languages}")
    print(f"Columns: {dataset.column_names}")
    
    # Print some statistics
    article_counts = {}
    for lang in languages:
        count = len([x for x in dataset if x["language"] == lang])
        article_counts[lang] = count
    
    print("\nArticle counts by language:")
    for lang, count in article_counts.items():
        print(f"  {lang}: {count}")
    
    # Print some sample titles for verification
    print("\nSample titles found:")
    for lang in languages:
        lang_articles = [x for x in dataset if x["language"] == lang]
        if lang_articles:
            print(f"  {lang}:")
            for i, article in enumerate(lang_articles[:3]):
                print(f"    {i+1}. {article['title']}")


if __name__ == "__main__":
    main()
