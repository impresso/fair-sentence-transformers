#!/usr/bin/env python3
"""
Script to process Wikipedia titles file and remove duplicates and ambiguous mappings.

This script processes a file containing Wikipedia title translations in the format:
"src_title ||| tgt_title"

Processing steps:
1. Remove trailing whitespaces from target titles
2. Remove duplicates resulting from whitespace removal
3. Remove all title pairs that are part of 1-n or n-1 relationships
   to ensure only unique mappings remain
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


def load_title_pairs(input_file: Path) -> List[Tuple[str, str]]:
    """Load title pairs from input file."""
    title_pairs = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            if " ||| " not in line:
                print(
                    f"Warning: Line {line_num} does not contain ' ||| ' separator: {line}"
                )
                continue

            parts = line.split(" ||| ", 1)  # Split only on first occurrence
            if len(parts) != 2:
                print(f"Warning: Line {line_num} has unexpected format: {line}")
                continue

            src_title, tgt_title = parts
            # Step 1: Remove trailing whitespaces from target title
            tgt_title = tgt_title.rstrip()
            title_pairs.append((src_title, tgt_title))

    return title_pairs


def remove_duplicates(title_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Remove duplicate title pairs."""
    # Step 2: Remove duplicates
    unique_pairs = list(dict.fromkeys(title_pairs))  # Preserves order

    duplicates_removed = len(title_pairs) - len(unique_pairs)
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate pairs")

    return unique_pairs


def find_ambiguous_mappings(title_pairs: List[Tuple[str, str]]) -> Set[str]:
    """
    Find source titles that are part of 1-n or n-1 relationships.

    Returns a set of source titles that should be excluded.
    """
    # Group by source title to find 1-n relationships
    src_to_tgt = defaultdict(set)
    # Group by target title to find n-1 relationships
    tgt_to_src = defaultdict(set)

    for src_title, tgt_title in title_pairs:
        src_to_tgt[src_title].add(tgt_title)
        tgt_to_src[tgt_title].add(src_title)

    ambiguous_src_titles = set()

    # Find 1-n relationships (one source maps to multiple targets)
    for src_title, tgt_titles in src_to_tgt.items():
        if len(tgt_titles) > 1:
            ambiguous_src_titles.add(src_title)
            print(f"1-n relationship found for '{src_title}': {sorted(tgt_titles)}")

    # Find n-1 relationships (multiple sources map to one target)
    for tgt_title, src_titles in tgt_to_src.items():
        if len(src_titles) > 1:
            ambiguous_src_titles.update(src_titles)
            print(f"n-1 relationship found for '{tgt_title}': {sorted(src_titles)}")

    return ambiguous_src_titles


def filter_unique_mappings(title_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Remove all title pairs that are part of ambiguous mappings."""
    # Step 3: Find and remove ambiguous mappings
    ambiguous_src_titles = find_ambiguous_mappings(title_pairs)

    if ambiguous_src_titles:
        print(
            f"Found {len(ambiguous_src_titles)} source titles with ambiguous mappings"
        )

    # Filter out ambiguous mappings
    unique_mappings = [
        (src_title, tgt_title)
        for src_title, tgt_title in title_pairs
        if src_title not in ambiguous_src_titles
    ]

    removed_count = len(title_pairs) - len(unique_mappings)
    if removed_count > 0:
        print(f"Removed {removed_count} pairs due to ambiguous mappings")

    return unique_mappings


def save_title_pairs(title_pairs: List[Tuple[str, str]], output_file: Path) -> None:
    """Save processed title pairs to output file."""
    with open(output_file, "w", encoding="utf-8") as f:
        for src_title, tgt_title in title_pairs:
            f.write(f"{src_title} ||| {tgt_title}\n")


def process_wikipedia_titles(input_file: Path, output_file: Path) -> None:
    """Main processing function."""
    print(f"Processing {input_file}")
    print(f"Output will be saved to {output_file}")

    # Load title pairs
    title_pairs = load_title_pairs(input_file)
    print(f"Loaded {len(title_pairs)} title pairs")

    # Remove duplicates
    title_pairs = remove_duplicates(title_pairs)
    print(f"After removing duplicates: {len(title_pairs)} title pairs")

    # Remove ambiguous mappings
    title_pairs = filter_unique_mappings(title_pairs)
    print(f"After removing ambiguous mappings: {len(title_pairs)} title pairs")

    # Save processed data
    save_title_pairs(title_pairs, output_file)
    print(f"Processed file saved to {output_file}")


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Process Wikipedia titles file to remove duplicates and ambiguous mappings"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input file path containing title pairs in format 'src_title ||| tgt_title'",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file path (default: adds '_processed' suffix to input filename)",
    )

    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: Input file {args.input_file} does not exist")
        return 1

    # Determine output file path
    if args.output:
        output_file = args.output
    else:
        # Add '_processed' suffix to input filename
        stem = args.input_file.stem
        suffix = args.input_file.suffix
        output_file = args.input_file.parent / f"{stem}_processed{suffix}"

    try:
        process_wikipedia_titles(args.input_file, output_file)
        return 0
    except Exception as e:
        print(f"Error processing file: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
