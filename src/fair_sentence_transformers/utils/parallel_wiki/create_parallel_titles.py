#!/usr/bin/env python3
"""
Script to create parallel Wikipedia titles file by performing inner join on English titles.

This script takes multiple Wikipedia title files (format: "src_title ||| tgt_title") 
and creates a single file containing only titles that exist in all input languages.

Usage:
    python create_parallel_titles.py <input_file1> <input_file2> ... [--output_dir OUTPUT_DIR]
    
Example:
    python create_parallel_titles.py de_titles_processed.txt it_titles_processed.txt ko_titles_processed.txt
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


def parse_title_file(file_path: str) -> Dict[str, str]:
    """
    Parse a title file and return a dictionary mapping English titles to source language titles.
    
    Args:
        file_path: Path to the title file
        
    Returns:
        Dictionary mapping English titles to source language titles
    """
    title_mapping = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(' ||| ')
            if len(parts) != 2:
                print(f"Warning: Skipping malformed line {line_num} in {file_path}: {line}")
                continue
                
            src_title, tgt_title = parts
            title_mapping[tgt_title] = src_title
    
    return title_mapping


def get_language_code(file_path: str) -> str:
    """
    Extract language code from file path.
    Assumes format like "xx_titles_processed.txt" where xx is the language code.
    
    Args:
        file_path: Path to the title file
        
    Returns:
        Language code (e.g., 'de', 'it', 'ko')
    """
    filename = os.path.basename(file_path)
    # Extract language code from filename like "de_titles_processed.txt"
    if '_titles_processed.txt' in filename:
        return filename.split('_titles_processed.txt')[0]
    elif '_titles.txt' in filename:
        return filename.split('_titles.txt')[0]
    else:
        # Fallback: use first part before underscore or dot
        return filename.split('_')[0].split('.')[0]


def create_parallel_titles(input_files: List[str], output_dir: str = None) -> str:
    """
    Create parallel titles file by performing inner join on English titles.
    
    Args:
        input_files: List of paths to input title files
        output_dir: Output directory (optional)
        
    Returns:
        Path to the created output file
    """
    if len(input_files) < 2:
        raise ValueError("At least 2 input files are required")
    
    # Parse all input files
    language_mappings = {}
    language_codes = []
    
    print(f"Processing {len(input_files)} input files...")
    
    for file_path in input_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        lang_code = get_language_code(file_path)
        language_codes.append(lang_code)
        
        print(f"  Parsing {file_path} (language: {lang_code})")
        language_mappings[lang_code] = parse_title_file(file_path)
        print(f"    Found {len(language_mappings[lang_code])} title pairs")
    
    # Find common English titles (inner join)
    print("\nFinding common English titles...")
    common_english_titles = None
    
    for lang_code in language_codes:
        english_titles = set(language_mappings[lang_code].keys())
        if common_english_titles is None:
            common_english_titles = english_titles
        else:
            common_english_titles = common_english_titles.intersection(english_titles)
        print(f"  After processing {lang_code}: {len(common_english_titles)} common titles")
    
    print(f"\nTotal common titles found: {len(common_english_titles)}")
    
    # Sort common titles for consistent output
    sorted_common_titles = sorted(common_english_titles)
    
    # Create output file
    if output_dir is None:
        # Use the directory of the first input file as default
        output_dir = os.path.dirname(input_files[0])
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    sorted_lang_codes = sorted(language_codes)
    output_filename = f"parallel_titles_en_{'_'.join(sorted_lang_codes)}.txt"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"\nWriting output to: {output_path}")
    
    # Write output file
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header line with language codes
        header = "en ||| " + " ||| ".join(sorted_lang_codes)
        f.write(header + "\n")
        
        # Write data lines
        for english_title in sorted_common_titles:
            line_parts = [english_title]
            for lang_code in sorted_lang_codes:
                src_title = language_mappings[lang_code][english_title]
                line_parts.append(src_title)
            
            line = " ||| ".join(line_parts)
            f.write(line + "\n")
    
    print(f"Successfully created parallel titles file with {len(sorted_common_titles)} entries")
    print(f"Languages included: {', '.join(sorted_lang_codes)}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create parallel Wikipedia titles file by performing inner join on English titles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process files in current directory
  python create_parallel_titles.py de_titles_processed.txt it_titles_processed.txt ko_titles_processed.txt
  
  # Process files with full paths and custom output directory
  python create_parallel_titles.py /path/to/de_titles_processed.txt /path/to/it_titles_processed.txt --output_dir /path/to/output
  
  # Process all *_titles_processed.txt files in a directory
  python create_parallel_titles.py /path/to/data/*_titles_processed.txt
        """
    )
    
    parser.add_argument(
        'input_files',
        nargs='+',
        help='Input title files (format: "src_title ||| tgt_title")'
    )
    
    parser.add_argument(
        '--output_dir',
        help='Output directory (default: directory of first input file)'
    )
    
    args = parser.parse_args()
    
    try:
        output_path = create_parallel_titles(args.input_files, args.output_dir)
        print(f"\n✅ Success! Output file created: {output_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
