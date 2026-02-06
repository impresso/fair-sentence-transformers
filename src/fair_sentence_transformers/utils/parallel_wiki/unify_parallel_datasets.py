#!/usr/bin/env python3
"""
Script to create a unified parallel dataset from two existing parquet datasets and a parallel IDs txt file.

This script creates a unified dataset where Wikipedia articles of the same topic/title are aligned across languages.
The output dataset contains one split per language, with samples aligned by "aligned_id".
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict


def load_parallel_ids(
    txt_file_path: str,
) -> Tuple[List[str], Dict[int, Dict[str, int]]]:
    """
    Load parallel IDs from the txt file.

    Args:
        txt_file_path: Path to the txt file containing parallel IDs

    Returns:
        Tuple of (language_codes, line_to_ids_mapping)
        where line_to_ids_mapping is {line_number: {lang: id, ...}}
    """
    with open(txt_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # First line contains language codes
    lang_codes = [lang.strip() for lang in lines[0].strip().split("|||")]

    # Parse parallel IDs for each line
    line_to_ids = {}
    for line_num, line in enumerate(lines[1:], start=1):
        id_strs = [id_str.strip() for id_str in line.strip().split("|||")]

        # Skip lines with empty values (missing articles in some languages)
        if any(not id_str for id_str in id_strs):
            continue

        try:
            ids = [int(id_str) for id_str in id_strs]
            line_to_ids[line_num] = dict(zip(lang_codes, ids))
        except ValueError:
            # Skip lines with invalid integer values
            continue

    return lang_codes, line_to_ids


def load_datasets(dataset1_path: str, dataset2_path: str) -> pd.DataFrame:
    """
    Load and combine the two parquet datasets.

    Args:
        dataset1_path: Path to first parquet dataset
        dataset2_path: Path to second parquet dataset

    Returns:
        Combined pandas DataFrame
    """
    df1 = pd.read_parquet(dataset1_path)
    df2 = pd.read_parquet(dataset2_path)

    # Combine datasets
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Convert id column to int for proper comparison
    combined_df["id"] = combined_df["id"].astype(int)

    # Create a lookup dictionary for fast access: (language, id) -> row
    lookup = {}
    for idx, row in combined_df.iterrows():
        key = (row["language"], row["id"])
        lookup[key] = {
            "wiki_pageid": row["id"],
            "language": row["language"],
            "url": row["url"],
            "title": row["title"],
            "text": row["text"],
        }

    return lookup


def create_unified_dataset(
    lang_codes: List[str],
    line_to_ids: Dict[int, Dict[str, int]],
    data_lookup: Dict[Tuple[str, int], Dict],
) -> DatasetDict:
    """
    Create the unified dataset with aligned splits.

    Args:
        lang_codes: List of language codes
        line_to_ids: Mapping from line number to language-specific IDs
        data_lookup: Lookup dictionary for dataset rows

    Returns:
        DatasetDict with one split per language
    """
    # Initialize data for each language split
    splits_data = {lang: [] for lang in lang_codes}

    # Process each line (aligned_id)
    for aligned_id, lang_to_id in line_to_ids.items():
        # Check if we have data for ALL languages for this aligned_id
        available_data = {}
        all_languages_present = True

        for lang in lang_codes:
            page_id = lang_to_id[lang]
            key = (lang, page_id)

            if key in data_lookup:
                available_data[lang] = data_lookup[key]
            else:
                all_languages_present = False
                break

        # Only add to dataset if ALL languages are present
        if all_languages_present:
            for lang in lang_codes:
                data = available_data[lang].copy()
                data["aligned_id"] = aligned_id
                splits_data[lang].append(data)

    # Convert to HuggingFace datasets
    dataset_dict = {}
    for lang in lang_codes:
        if splits_data[lang]:  # Only create split if we have data
            # Add enumerated "id" feature to each sample
            for i, sample in enumerate(splits_data[lang]):
                sample["id"] = f"{i}{lang}"
            dataset_dict[lang] = Dataset.from_list(splits_data[lang])

    return DatasetDict(dataset_dict)


def verify_alignment(dataset_dict: DatasetDict) -> bool:
    """
    Verify that all splits are properly aligned.

    Args:
        dataset_dict: The unified dataset

    Returns:
        True if properly aligned, False otherwise
    """
    splits = list(dataset_dict.keys())
    if len(splits) < 2:
        return True

    # Check that all splits have the same length
    lengths = [len(dataset_dict[split]) for split in splits]
    if len(set(lengths)) > 1:
        print(f"ERROR: Splits have different lengths: {dict(zip(splits, lengths))}")
        return False

    # Check that aligned_ids match across all splits for each position
    length = lengths[0]
    for i in range(min(100, length)):  # Check first 100 samples
        aligned_ids = [dataset_dict[split][i]["aligned_id"] for split in splits]
        if len(set(aligned_ids)) > 1:
            print(
                f"ERROR: Misaligned samples at position {i}: {dict(zip(splits, aligned_ids))}"
            )
            return False

    print(
        f"SUCCESS: All {len(splits)} splits are properly aligned with {length} samples each"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Create unified parallel dataset")
    parser.add_argument(
        "--dataset1", required=True, help="Path to first parquet dataset"
    )
    parser.add_argument(
        "--dataset2", required=True, help="Path to second parquet dataset"
    )
    parser.add_argument(
        "--parallel_ids", required=True, help="Path to parallel IDs txt file"
    )
    parser.add_argument(
        "--output_dir", help="Output directory (default: same as first dataset)"
    )
    parser.add_argument(
        "--output_name",
        help="Output dataset name (default: unified_<first_dataset_name>)",
    )

    args = parser.parse_args()

    # Set default output directory and name
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.dataset1)

    if args.output_name is None:
        dataset1_name = Path(args.dataset1).stem
        args.output_name = f"unified_{dataset1_name}"

    output_path = os.path.join(args.output_dir, args.output_name)

    print(f"Loading parallel IDs from: {args.parallel_ids}")
    lang_codes, line_to_ids = load_parallel_ids(args.parallel_ids)
    print(f"Found {len(lang_codes)} languages: {lang_codes}")
    print(f"Found {len(line_to_ids)} parallel article groups")

    print(f"\nLoading datasets:")
    print(f"  Dataset 1: {args.dataset1}")
    print(f"  Dataset 2: {args.dataset2}")
    data_lookup = load_datasets(args.dataset1, args.dataset2)
    print(f"Loaded {len(data_lookup)} unique (language, id) pairs")

    print(f"\nCreating unified dataset...")
    dataset_dict = create_unified_dataset(lang_codes, line_to_ids, data_lookup)

    print(f"\nDataset splits created:")
    for split, dataset in dataset_dict.items():
        print(f"  {split}: {len(dataset)} samples")

    print(f"\nVerifying alignment...")
    if verify_alignment(dataset_dict):
        print(f"\nSaving unified dataset to: {output_path}")
        dataset_dict.save_to_disk(output_path)
        print("Dataset saved successfully!")

        # Print sample from each split to verify
        print(f"\nSample verification (first sample from each split):")
        for split in dataset_dict.keys():
            sample = dataset_dict[split][0]
            print(
                f"  {split}: aligned_id={sample['aligned_id']}, "
                f"wiki_pageid={sample['wiki_pageid']}, "
                f"title='{sample['title'][:50]}...'"
            )
    else:
        print("ERROR: Dataset alignment verification failed!")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
