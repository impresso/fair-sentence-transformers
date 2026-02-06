#!/usr/bin/env python
"""
Parallel Indices Creation Script for fair_sentence_transformers

This script creates concat_indices and standalone_indices for parallel datasets
with multiple languages. It filters segments based on token length constraints
and ensures parallel segments meet length similarity requirements.

Usage:
    python create_parallel_indices.py --config PATH_TO_CONFIG_FILE
    or
    python create_parallel_indices.py --dataset_dir PATH --source_lang LANG [other args]
"""

import os
import sys
import json
import argparse
import itertools
import math
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        return json.load(f)


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """
    Load metadata from a JSON file.

    Args:
        metadata_path: Path to the metadata JSON file

    Returns:
        Metadata dictionary
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_parallel_indices(
    dataset_dir: str,
    source_lang: str,
    target_langs: Optional[List[str]] = None,
    concat_size: int = 2,
    sample_size: int = 100,
    max_total_length: Optional[int] = None,
    source_segment_range: Optional[Tuple[int, int]] = None,
    source_target_segment_diffs: Optional[List[float]] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Create concat_indices and standalone_indices for parallel datasets.

    Args:
        dataset_dir: Path to the directory containing tokenized dataset
        source_lang: Source language code
        target_langs: Optional list of target language codes
        concat_size: Number of segments to concatenate
        sample_size: Number of concatenated samples to create
        max_total_length: Maximum total token length for concatenated documents
        source_segment_range: Tuple defining acceptable token length range for source segments
        source_target_segment_diffs: List of fractions defining acceptable length differences
                                   between source and target segments for each target language

    Returns:
        A tuple containing:
        - A nested list where each inner list contains dataset indices for concatenation
        - A sorted list of all unique indices used
    """
    if concat_size < 2:
        raise ValueError(
            f"Concatenation size must be at least 2, but got {concat_size}"
        )

    if sample_size < 1:
        raise ValueError(f"Sample size must be at least 1, but got {sample_size}")

    # Ensure target_langs is a list
    if target_langs is None:
        target_langs = []

    # Validate source_target_segment_diffs length
    if source_target_segment_diffs is not None and len(
        source_target_segment_diffs
    ) != len(target_langs):
        raise ValueError(
            f"Length of source_target_segment_diffs ({len(source_target_segment_diffs)}) "
            f"must match number of target languages ({len(target_langs)})"
        )

    # Load source language metadata
    source_metadata_path = os.path.join(dataset_dir, source_lang, "metadata.json")
    if not os.path.exists(source_metadata_path):
        raise FileNotFoundError(
            f"Source metadata file not found: {source_metadata_path}"
        )

    source_metadata = load_metadata(source_metadata_path)
    print(
        f"Loaded source metadata for '{source_lang}': {len(source_metadata)} documents"
    )

    # Load target language metadata
    target_metadata = {}
    for target_lang in target_langs:
        target_metadata_path = os.path.join(dataset_dir, target_lang, "metadata.json")
        if not os.path.exists(target_metadata_path):
            raise FileNotFoundError(
                f"Target metadata file not found: {target_metadata_path}"
            )

        target_metadata[target_lang] = load_metadata(target_metadata_path)
        print(
            f"Loaded target metadata for '{target_lang}': {len(target_metadata[target_lang])} documents"
        )

    # Extract source language information
    source_indices = []
    source_lengths = {}

    for doc_id, doc_info in source_metadata.items():
        idx = doc_info["dataset_idx"]
        length = doc_info["token_length"]
        source_indices.append(idx)
        source_lengths[idx] = length

    # Extract target language information
    target_lengths = {}
    for target_lang in target_langs:
        target_lengths[target_lang] = {}
        for doc_id, doc_info in target_metadata[target_lang].items():
            idx = doc_info["dataset_idx"]
            length = doc_info["token_length"]
            target_lengths[target_lang][idx] = length

    print(f"Total source documents: {len(source_indices)}")

    # Filter source segments based on source_segment_range
    if source_segment_range is not None:
        min_length, max_length = source_segment_range
        filtered_source_indices = [
            idx
            for idx in source_indices
            if min_length <= source_lengths[idx] <= max_length
        ]
        print(
            f"After source length filtering ({min_length}-{max_length}): {len(filtered_source_indices)} documents"
        )
    else:
        filtered_source_indices = source_indices.copy()

    # Apply target language length difference constraints
    if target_langs and source_target_segment_diffs is not None:
        valid_indices = []
        for idx in filtered_source_indices:
            source_length = source_lengths[idx]
            is_valid = True

            for target_lang, diff_threshold in zip(
                target_langs, source_target_segment_diffs
            ):
                if idx not in target_lengths[target_lang]:
                    is_valid = False
                    break

                target_length = target_lengths[target_lang][idx]
                min_acceptable = source_length * (1 - diff_threshold)
                max_acceptable = source_length * (1 + diff_threshold)

                if not (min_acceptable <= target_length <= max_acceptable):
                    is_valid = False
                    break

            if is_valid:
                valid_indices.append(idx)

        filtered_source_indices = valid_indices
        print(
            f"After target length difference filtering: {len(filtered_source_indices)} documents"
        )

    # Check if we have enough documents
    if len(filtered_source_indices) < concat_size:
        raise ValueError(
            f"After filtering, only {len(filtered_source_indices)} documents remain, "
            f"but {concat_size} are required for concatenation"
        )

    # Adjust sample_size to be a multiple of factorial(concat_size) for permutations
    factorial_size = math.factorial(concat_size)
    if sample_size % factorial_size != 0:
        original_sample_size = sample_size
        sample_size = ((sample_size // factorial_size) + 1) * factorial_size
        print(
            f"Adjusted sample_size from {original_sample_size} to {sample_size} "
            f"to ensure it's a multiple of {factorial_size} (factorial of concat_size {concat_size})"
        )

    # Generate concatenation indices using permutations strategy
    base_sample_count = sample_size // factorial_size
    concatenation_indices = []
    used_indices = set()
    added_base_combinations = set()

    samples_created = 0
    max_attempts = base_sample_count * 10
    attempts = 0

    print(
        f"Generating {base_sample_count} base combinations with {factorial_size} permutations each..."
    )

    while samples_created < base_sample_count and attempts < max_attempts:
        attempts += 1

        # Select random documents for concatenation
        if len(filtered_source_indices) >= concat_size:
            selected_indices = random.sample(filtered_source_indices, concat_size)
        else:
            # If we don't have enough unique documents, allow repetition
            selected_indices = [
                random.choice(filtered_source_indices) for _ in range(concat_size)
            ]

        # Check total length constraint if specified
        if max_total_length is not None:
            total_length = sum(source_lengths[idx] for idx in selected_indices)
            if total_length > max_total_length:
                continue

        # Create canonical representation to avoid duplicate sets
        canonical_tuple = tuple(sorted(selected_indices))

        # Only add if we haven't already created permutations for this set of documents
        if canonical_tuple not in added_base_combinations:
            # Generate all permutations
            for permutation in itertools.permutations(selected_indices):
                permutation_list = list(permutation)
                concatenation_indices.append(permutation_list)

            # Track that we've used this set of documents
            added_base_combinations.add(canonical_tuple)
            used_indices.update(selected_indices)
            samples_created += 1

    if samples_created < base_sample_count:
        actual_samples = samples_created * factorial_size
        print(
            f"Warning: Could only generate {actual_samples} samples "
            f"(instead of requested {sample_size}) before reaching max attempts."
        )

    # Convert the set of used indices to a sorted list
    standalone_indices = sorted(list(used_indices))

    print(f"Generated {len(concatenation_indices)} concatenation samples")
    print(f"Generated {len(standalone_indices)} standalone indices")

    return concatenation_indices, standalone_indices


def save_indices(
    concat_indices: List[List[int]],
    standalone_indices: List[int],
    output_path: str,
    config: Dict[str, Any],
) -> None:
    """
    Save the generated indices and configuration to a JSON file.

    Args:
        concat_indices: List of concatenation index lists
        standalone_indices: List of standalone indices
        output_path: Path to save the output JSON file
        config: Configuration used to generate the indices
    """
    output_data = {
        "concat_indices": concat_indices,
        "standalone_indices": standalone_indices,
        "generation_config": config,
        "stats": {
            "num_concat_samples": len(concat_indices),
            "num_standalone_indices": len(standalone_indices),
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Indices saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create parallel indices for fair_sentence_transformers"
    )

    # Config file option
    parser.add_argument("--config", help="Path to configuration JSON file")

    # Individual parameters
    parser.add_argument("--dataset_dir", help="Path to tokenized dataset directory")
    parser.add_argument("--source_lang", help="Source language code")
    parser.add_argument("--target_langs", nargs="*", help="Target language codes")
    parser.add_argument(
        "--concat_size", type=int, default=2, help="Number of segments to concatenate"
    )
    parser.add_argument(
        "--sample_size", type=int, default=100, help="Number of samples to create"
    )
    parser.add_argument(
        "--max_total_length", type=int, help="Maximum total token length"
    )
    parser.add_argument(
        "--source_segment_range",
        nargs=2,
        type=int,
        metavar=("MIN", "MAX"),
        help="Acceptable token length range for source segments",
    )
    parser.add_argument(
        "--source_target_segment_diffs",
        nargs="*",
        type=float,
        help="Acceptable length difference fractions for target languages",
    )
    parser.add_argument("--output_path", help="Path to save the generated indices")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = load_config(args.config)

        # Extract parameters from config
        dataset_dir = config["dataset_dir"]
        source_lang = config["source_lang"]
        target_langs = config.get("target_langs", [])
        concat_size = config.get("concat_size", 2)
        sample_size = config.get("sample_size", 100)
        max_total_length = config.get("max_total_length")
        source_segment_range = config.get("source_segment_range")
        if source_segment_range:
            source_segment_range = tuple(source_segment_range)
        source_target_segment_diffs = config.get("source_target_segment_diffs")
        output_path = config.get("output_path")

    else:
        # Use command line arguments
        if not args.dataset_dir or not args.source_lang:
            parser.error(
                "--dataset_dir and --source_lang are required when not using --config"
            )

        config = {}
        dataset_dir = args.dataset_dir
        source_lang = args.source_lang
        target_langs = args.target_langs or []
        concat_size = args.concat_size
        sample_size = args.sample_size
        max_total_length = args.max_total_length
        source_segment_range = (
            tuple(args.source_segment_range) if args.source_segment_range else None
        )
        source_target_segment_diffs = args.source_target_segment_diffs
        output_path = args.output_path

    # Validate output path
    if not output_path:
        # Generate default output path
        base_name = f"indices_{source_lang}"
        if target_langs:
            base_name += f"_{'_'.join(target_langs)}"
        base_name += f"_concat{concat_size}_samples{sample_size}.json"
        output_path = os.path.join(os.path.dirname(dataset_dir), base_name)

    # Create the configuration dictionary for saving
    generation_config = {
        "dataset_dir": dataset_dir,
        "source_lang": source_lang,
        "target_langs": target_langs,
        "concat_size": concat_size,
        "sample_size": sample_size,
        "max_total_length": max_total_length,
        "source_segment_range": source_segment_range,
        "source_target_segment_diffs": source_target_segment_diffs,
    }

    print("Creating parallel indices with configuration:")
    print(json.dumps(generation_config, indent=2))

    # Generate indices
    concat_indices, standalone_indices = create_parallel_indices(
        dataset_dir=dataset_dir,
        source_lang=source_lang,
        target_langs=target_langs,
        concat_size=concat_size,
        sample_size=sample_size,
        max_total_length=max_total_length,
        source_segment_range=source_segment_range,
        source_target_segment_diffs=source_target_segment_diffs,
    )

    # Save indices
    save_indices(concat_indices, standalone_indices, output_path, generation_config)


if __name__ == "__main__":
    main()
