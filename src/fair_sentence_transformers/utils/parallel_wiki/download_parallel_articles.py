#!/usr/bin/env python3
"""
Script to download parallel Wikipedia articles based on parallel pageids and/or titles files.

This script takes .txt files with parallel Wikipedia pageids and/or titles and downloads the
corresponding articles from Hugging Face Wikipedia datasets in streaming mode.
It creates a unified dataset with all languages and additional metadata columns.

For specified languages, title-based matching can be used instead of ID-based matching.
This is useful when ID matching doesn't work well for certain languages (e.g., English).

Usage:
    python download_parallel_articles.py <input_pageids_file> [--titles_file <titles_file>] [--title_match_languages <lang1> <lang2> ...]

Examples:
    python download_parallel_articles.py /path/to/parallel_ids_en_de_hi_it_ko.txt
    python download_parallel_articles.py /path/to/parallel_ids_en_de_hi_it_ko.txt --titles_file /path/to/parallel_titles_en_de_hi_it_ko.txt --title_match_languages en
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import sys

from datasets import Dataset, load_dataset
from tqdm import tqdm


def parse_combined_files(
    pageids_file_path: str,
    titles_file_path: str = None,
    all_languages: List[str] = None,
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, List[str]], List[int]]:
    """
    Parse both pageids and titles files together to ensure consistent article grouping.

    Args:
        pageids_file_path: Path to the parallel pageids file
        titles_file_path: Path to the parallel titles file (optional)
        all_languages: List of all language codes (optional, inferred from pageids file if not provided)

    Returns:
        Tuple of (languages, pageids_by_language, titles_by_language, valid_article_indices)
    """
    # Parse pageids file first
    with open(pageids_file_path, "r", encoding="utf-8") as f:
        pageids_lines = f.readlines()

    # Parse header to get language codes
    pageids_header = pageids_lines[0].strip()
    languages = [lang.strip() for lang in pageids_header.split("|||")]

    if all_languages and set(all_languages) != set(languages):
        raise ValueError(
            f"Language mismatch between files. Expected: {all_languages}, Found: {languages}"
        )

    # Parse titles file if provided
    titles_lines = None
    if titles_file_path:
        with open(titles_file_path, "r", encoding="utf-8") as f:
            titles_lines = f.readlines()

        # Validate titles file header
        titles_header = titles_lines[0].strip()
        titles_languages = [lang.strip() for lang in titles_header.split("|||")]

        if set(titles_languages) != set(languages):
            raise ValueError(
                f"Language mismatch between pageids and titles files. Pageids: {languages}, Titles: {titles_languages}"
            )

        # Reorder titles languages to match pageids order
        lang_indices = {lang: titles_languages.index(lang) for lang in languages}

    # Process both files line by line to ensure consistent filtering
    pageids_by_language = {lang: [] for lang in languages}
    titles_by_language = {lang: [] for lang in languages} if titles_lines else None
    valid_article_indices = []

    max_lines = len(pageids_lines)
    if titles_lines and len(titles_lines) != max_lines:
        print(
            f"Warning: Files have different number of lines. Pageids: {max_lines}, Titles: {len(titles_lines)}"
        )
        max_lines = min(max_lines, len(titles_lines))

    for line_idx in range(1, max_lines):  # Skip header
        pageids_line = pageids_lines[line_idx].strip()
        titles_line = titles_lines[line_idx].strip() if titles_lines else None

        if not pageids_line or (titles_lines and not titles_line):
            continue

        # Parse pageids
        pageids = [pageid.strip() for pageid in pageids_line.split("|||")]
        if len(pageids) != len(languages):
            print(
                f"Warning: Skipping malformed pageids line {line_idx + 1}: {pageids_line}"
            )
            continue

        # Parse titles if available
        titles = None
        if titles_lines:
            titles = [title.strip() for title in titles_line.split("|||")]
            if len(titles) != len(titles_languages):
                print(
                    f"Warning: Skipping malformed titles line {line_idx + 1}: {titles_line}"
                )
                continue
            # Reorder to match pageids language order
            titles = [titles[lang_indices[lang]] for lang in languages]

        # Check if all pageids are present and valid
        pageids_valid = all(pageid for pageid in pageids)
        if pageids_valid:
            try:
                [int(pageid) for pageid in pageids]
            except ValueError:
                print(
                    f"Warning: Skipping line {line_idx + 1} with invalid pageids: {pageids_line}"
                )
                continue

        # Check if all titles are present (if titles file provided)
        titles_valid = True
        if titles_lines:
            titles_valid = all(title for title in titles)

        # Only include articles that are valid in both files (if both provided)
        if pageids_valid and titles_valid:
            for lang, pageid in zip(languages, pageids):
                pageids_by_language[lang].append(pageid)

            if titles_lines:
                for lang, title in zip(languages, titles):
                    titles_by_language[lang].append(title)

            valid_article_indices.append(line_idx - 1)  # 0-based index
        else:
            missing_info = []
            if not pageids_valid:
                missing_info.append("pageids")
            if titles_lines and not titles_valid:
                missing_info.append("titles")
            print(
                f"Warning: Skipping line {line_idx + 1} with missing {'/'.join(missing_info)}"
            )

    print(f"Loaded {len(valid_article_indices)} valid parallel articles")

    return languages, pageids_by_language, titles_by_language, valid_article_indices


def download_articles_for_language(
    language: str, identifiers: List[str], use_title_matching: bool = False
) -> List[Dict]:
    """
    Download Wikipedia articles for a specific language using streaming.

    Args:
        language: Language code (e.g., 'en', 'de', 'hi', 'it', 'ko')
        identifiers: List of pageids (for ID matching) or titles (for title matching) to collect
        use_title_matching: If True, match by title; if False, match by ID

    Returns:
        List of article dictionaries
    """
    print(f"Loading Wikipedia dataset for language: {language}")

    # Load dataset in streaming mode
    dataset = load_dataset(
        "wikimedia/wikipedia", f"20231101.{language}", streaming=True, split="train"
    )

    articles = []
    identifiers_set = set(identifiers)
    found_identifiers = set()

    if use_title_matching:
        print(f"Processing {len(identifiers)} titles for {language} (title matching)")
        print("Streaming through entire dataset to find matching titles...")

        with tqdm(desc=f"Processing {language} (titles)", unit="articles") as pbar:
            for article in dataset:
                article_title = article["title"]

                # Check if this article title matches any of our target titles
                if article_title in identifiers_set:
                    articles.append(
                        {
                            "language": language,
                            "id": article["id"],
                            "url": article["url"],
                            "title": article["title"],
                            "text": article["text"],
                        }
                    )
                    found_identifiers.add(article_title)
                    pbar.update(1)

                    # If we've found all titles, we can break early
                    if len(found_identifiers) == len(identifiers_set):
                        break

                # Update progress bar periodically (every 10000 articles)
                if pbar.n % 10000 == 0:
                    pbar.set_postfix(
                        {
                            "found": len(found_identifiers),
                            "target": len(identifiers_set),
                        }
                    )

        # Report missing titles
        missing_identifiers = identifiers_set - found_identifiers
        if missing_identifiers:
            print(
                f"Warning: {len(missing_identifiers)} titles not found for {language}:"
            )
            for title in sorted(missing_identifiers):
                print(f"  - {title}")

    else:
        # Original ID-based matching with optimization
        pageids = identifiers  # For clarity
        pageid_idx = 0
        current_target_pageid = int(pageids[0]) if pageids else None

        print(f"Processing {len(pageids)} pageids for {language} (ID matching)")

        with tqdm(desc=f"Processing {language} (IDs)", unit="articles") as pbar:
            for article in dataset:
                article_id = article["id"]
                article_id_int = int(article_id)

                # Skip articles with IDs lower than our current target
                if (
                    current_target_pageid is not None
                    and article_id_int < current_target_pageid
                ):
                    continue

                # Check if this article ID matches any of our target pageids
                if article_id in identifiers_set:
                    articles.append(
                        {
                            "language": language,
                            "id": article_id,
                            "url": article["url"],
                            "title": article["title"],
                            "text": article["text"],
                        }
                    )
                    found_identifiers.add(article_id)
                    pbar.update(1)

                    # Move to next target pageid
                    pageid_idx += 1
                    if pageid_idx < len(pageids):
                        current_target_pageid = int(pageids[pageid_idx])
                    else:
                        # We've found all articles, break early
                        break

                # If we've passed the current target, move to the next one
                elif (
                    current_target_pageid is not None
                    and article_id_int > current_target_pageid
                ):
                    pageid_idx += 1
                    if pageid_idx < len(pageids):
                        current_target_pageid = int(pageids[pageid_idx])
                        # Check if the current article matches the new target
                        if (
                            article_id == pageids[pageid_idx - 1]
                        ):  # Off by one correction
                            articles.append(
                                {
                                    "language": language,
                                    "id": article_id,
                                    "url": article["url"],
                                    "title": article["title"],
                                    "text": article["text"],
                                }
                            )
                            found_identifiers.add(article_id)
                            pbar.update(1)
                    else:
                        # No more targets, break
                        break

        # Report missing pageids
        missing_identifiers = identifiers_set - found_identifiers
        if missing_identifiers:
            print(
                f"Warning: {len(missing_identifiers)} pageids not found for {language}:"
            )
            for pageid in sorted(missing_identifiers, key=lambda x: int(x)):
                print(f"  - {pageid}")

    print(f"Found {len(articles)} articles for {language}")
    return articles


def create_unified_dataset(
    languages: List[str],
    pageids_by_language: Dict[str, List[str]],
    titles_by_language: Dict[str, List[str]] = None,
    title_match_languages: List[str] = None,
) -> Dataset:
    """
    Create a unified dataset with all languages and additional metadata columns.

    Args:
        languages: List of language codes
        pageids_by_language: Dictionary mapping language codes to pageids
        titles_by_language: Dictionary mapping language codes to titles (optional)
        title_match_languages: List of languages that should use title matching

    Returns:
        Unified Dataset object
    """
    if title_match_languages is None:
        title_match_languages = []

    # For ID-based matching, we need to sort pageids for optimization
    # But we need to maintain the article grouping, so we sort all languages together
    sorted_pageids_by_language = {}
    sorted_titles_by_language = {}
    article_id_mapping = {}  # Maps original index to new sorted index

    if any(lang not in title_match_languages for lang in languages):
        # Create tuples of (article_index, pageids_for_all_langs, titles_for_all_langs)
        article_tuples = []
        for i in range(len(pageids_by_language[languages[0]])):
            pageids_tuple = tuple(pageids_by_language[lang][i] for lang in languages)
            titles_tuple = tuple(
                titles_by_language[lang][i] if titles_by_language else None
                for lang in languages
            )
            article_tuples.append((i, pageids_tuple, titles_tuple))

        # Sort by the first language's pageid (which will be used for ID optimization)
        if languages[0] not in title_match_languages:
            article_tuples.sort(key=lambda x: int(x[1][0]))

        # Rebuild the dictionaries in sorted order
        for lang_idx, lang in enumerate(languages):
            sorted_pageids_by_language[lang] = []
            if titles_by_language:
                sorted_titles_by_language[lang] = []

            for new_idx, (orig_idx, pageids_tuple, titles_tuple) in enumerate(
                article_tuples
            ):
                sorted_pageids_by_language[lang].append(pageids_tuple[lang_idx])
                if titles_by_language:
                    sorted_titles_by_language[lang].append(titles_tuple[lang_idx])
                article_id_mapping[new_idx] = orig_idx
    else:
        # All languages use title matching, no need to sort
        sorted_pageids_by_language = pageids_by_language
        sorted_titles_by_language = titles_by_language if titles_by_language else {}
        article_id_mapping = {
            i: i for i in range(len(pageids_by_language[languages[0]]))
        }

    all_articles = []

    # Download articles for each language
    articles_by_language = {}
    for language in languages:
        use_title_matching = language in title_match_languages

        if use_title_matching:
            if titles_by_language is None or language not in titles_by_language:
                raise ValueError(
                    f"Title matching requested for {language} but no titles provided"
                )
            identifiers = sorted_titles_by_language[language]
        else:
            identifiers = sorted_pageids_by_language[language]

        articles_by_language[language] = download_articles_for_language(
            language, identifiers, use_title_matching
        )

    # Create mapping from identifier to article_id (parallel article group index)
    # Use the original article indices (before sorting)
    identifier_to_article_id = {}

    for sorted_idx in range(len(sorted_pageids_by_language[languages[0]])):
        original_article_id = article_id_mapping[sorted_idx]

        for lang in languages:
            if lang in title_match_languages and titles_by_language:
                identifier = sorted_titles_by_language[lang][sorted_idx]
                key = f"{lang}_title_{identifier}"
            else:
                identifier = sorted_pageids_by_language[lang][sorted_idx]
                key = f"{lang}_id_{identifier}"
            identifier_to_article_id[key] = original_article_id

    # Combine all articles with metadata
    idx = 0
    for language in languages:
        use_title_matching = language in title_match_languages

        for article in articles_by_language[language]:
            # Find the article_id for this article
            if use_title_matching:
                key = f"{language}_title_{article['title']}"
            else:
                key = f"{language}_id_{article['id']}"

            article_id = identifier_to_article_id.get(key, -1)

            article_with_metadata = {
                "idx": idx,
                "article_id": article_id,
                "language": article["language"],
                "id": article["id"],
                "url": article["url"],
                "title": article["title"],
                "text": article["text"],
            }
            all_articles.append(article_with_metadata)
            idx += 1

    # Create dataset
    return Dataset.from_list(all_articles)


def main():
    parser = argparse.ArgumentParser(
        description="Download parallel Wikipedia articles from pageids and/or titles files"
    )
    parser.add_argument("input_file", help="Path to the parallel pageids .txt file")
    parser.add_argument(
        "--titles_file",
        help="Path to the parallel titles .txt file (optional)",
        default=None,
    )
    parser.add_argument(
        "--title_match_languages",
        nargs="*",
        help="Language codes that should use title matching instead of ID matching",
        default=[],
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory (default: same directory as input file)",
        default=None,
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        sys.exit(1)

    # Validate titles file if provided
    titles_path = None
    if args.titles_file:
        titles_path = Path(args.titles_file)
        if not titles_path.exists():
            print(f"Error: Titles file {titles_path} does not exist")
            sys.exit(1)

    # Check if title matching is requested but no titles file provided
    if args.title_match_languages and not args.titles_file:
        print("Error: Title matching languages specified but no titles file provided")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse both files together to ensure consistent article grouping
    print(f"Parsing input files...")
    print(f"Pageids file: {input_path}")
    if args.titles_file:
        print(f"Titles file: {titles_path}")

    languages, pageids_by_language, titles_by_language, valid_article_indices = (
        parse_combined_files(
            str(input_path), str(titles_path) if args.titles_file else None
        )
    )

    print(f"Found languages: {languages}")
    print(f"Number of parallel articles: {len(pageids_by_language[languages[0]])}")

    # Validate title matching languages
    if args.title_match_languages:
        for lang in args.title_match_languages:
            if lang not in languages:
                print(
                    f"Error: Language '{lang}' not found in input files. Available: {languages}"
                )
                sys.exit(1)
        print(f"Using title matching for: {args.title_match_languages}")
        print(
            f"Using ID matching for: {[lang for lang in languages if lang not in args.title_match_languages]}"
        )

    # Create unified dataset
    print("Creating unified dataset...")
    dataset = create_unified_dataset(
        languages, pageids_by_language, titles_by_language, args.title_match_languages
    )

    # Save dataset
    output_name = input_path.stem.replace("parallel_ids", "parallel_articles")
    if args.title_match_languages:
        title_langs_str = "_".join(sorted(args.title_match_languages))
        output_name += f"_titleMatch_{title_langs_str}"
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
        matching_type = "title" if lang in args.title_match_languages else "ID"
        print(f"  {lang}: {count} ({matching_type} matching)")

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
