#!/usr/bin/env python3
"""
Script to convert Wikipedia parallel titles to page IDs.

This script reads a file containing parallel Wikipedia titles across multiple languages
and converts them to their corresponding page IDs using the Wikipedia API.

Input format:
en ||| de ||| hi ||| it ||| ko
'Amran Governorate ||| Gouvernement ʿAmrān ||| अमरान प्रान्त ||| Governatorato di 'Amran ||| 암란주
...

Output format:
en ||| de ||| hi ||| it ||| ko
12345 ||| 67890 ||| 11111 ||| 22222 ||| 33333
...

Usage:
    python title_to_pageid.py /path/to/parallel_titles_en_de_hi_it_ko.txt
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
import argparse

# Add the parent directory to the path to import the translator
sys.path.append(str(Path(__file__).parent.parent))
from wikipedia_id_translator import WikipediaCuridTranslator


class ParallelTitleToPageIDConverter:
    def __init__(self, languages: List[str]):
        """
        Initialize the converter with a list of language codes.

        Args:
            languages: List of language codes (e.g., ['en', 'de', 'hi', 'it', 'ko'])
        """
        self.languages = languages
        # Create a translator instance (we'll use the _batch_get_pageids_from_titles method directly)
        self.translator = WikipediaCuridTranslator(
            "en", "de"
        )  # source/target don't matter for this use case

    def parse_input_file(self, filepath: str) -> List[List[str]]:
        """
        Parse the input file and return a list of title rows.

        Args:
            filepath: Path to the input file

        Returns:
            List of lists, where each inner list contains titles for each language
        """
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Skip the header line (language codes)
        title_lines = lines[1:]

        title_rows = []
        for line in title_lines:
            line = line.strip()
            if line:
                titles = [title.strip() for title in line.split(" ||| ")]
                if len(titles) == len(self.languages):
                    title_rows.append(titles)
                else:
                    print(f"Warning: Skipping malformed line: {line}")

        return title_rows

    def get_pageids_for_language(
        self, titles: List[str], lang: str
    ) -> Dict[str, Optional[int]]:
        """
        Get page IDs for a list of titles in a specific language.

        Args:
            titles: List of Wikipedia titles
            lang: Language code (e.g., 'en', 'de')

        Returns:
            Dictionary mapping title to page ID (or None if not found)
        """
        return self.translator._batch_get_pageids_from_titles(titles, lang)

    def convert_titles_to_pageids(
        self, title_rows: List[List[str]]
    ) -> List[List[Optional[int]]]:
        """
        Convert title rows to page ID rows.

        Args:
            title_rows: List of title rows, each containing titles for all languages

        Returns:
            List of page ID rows, each containing page IDs for all languages
        """
        # Collect all titles for each language
        titles_by_lang = {lang: [] for lang in self.languages}
        for row in title_rows:
            for i, title in enumerate(row):
                titles_by_lang[self.languages[i]].append(title)

        # Get page IDs for each language
        pageids_by_lang = {}
        for lang in self.languages:
            print(f"Getting page IDs for {lang}...")
            unique_titles = list(set(titles_by_lang[lang]))
            pageids_by_lang[lang] = self.get_pageids_for_language(unique_titles, lang)

        # Convert title rows to page ID rows
        pageid_rows = []
        for row in title_rows:
            pageid_row = []
            for i, title in enumerate(row):
                lang = self.languages[i]
                pageid = pageids_by_lang[lang].get(title)
                pageid_row.append(pageid)
            pageid_rows.append(pageid_row)

        return pageid_rows

    def write_output_file(
        self, pageid_rows: List[List[Optional[int]]], output_filepath: str
    ):
        """
        Write page ID rows to the output file.

        Args:
            pageid_rows: List of page ID rows
            output_filepath: Path to the output file
        """
        with open(output_filepath, "w", encoding="utf-8") as f:
            # Write header
            f.write(" ||| ".join(self.languages) + "\n")

            # Write page ID rows
            for row in pageid_rows:
                formatted_row = []
                for pageid in row:
                    if pageid is not None:
                        formatted_row.append(str(pageid))
                    else:
                        formatted_row.append("")  # Empty string for missing page IDs
                f.write(" ||| ".join(formatted_row) + "\n")

    def convert_file(self, input_filepath: str) -> str:
        """
        Convert a titles file to a page IDs file.

        Args:
            input_filepath: Path to the input file

        Returns:
            Path to the output file
        """
        print(f"Reading titles from: {input_filepath}")
        title_rows = self.parse_input_file(input_filepath)
        print(f"Found {len(title_rows)} title rows")

        print("Converting titles to page IDs...")
        pageid_rows = self.convert_titles_to_pageids(title_rows)

        # Generate output filepath
        input_path = Path(input_filepath)
        output_filename = input_path.name.replace("parallel_titles_", "parallel_ids_")
        output_filepath = input_path.parent / output_filename

        print(f"Writing page IDs to: {output_filepath}")
        self.write_output_file(pageid_rows, str(output_filepath))

        # Print statistics
        total_cells = len(pageid_rows) * len(self.languages)
        missing_cells = sum(
            1 for row in pageid_rows for pageid in row if pageid is None
        )
        success_rate = (total_cells - missing_cells) / total_cells * 100
        print(f"Conversion complete!")
        print(
            f"Success rate: {success_rate:.2f}% ({total_cells - missing_cells}/{total_cells} page IDs found)"
        )

        return str(output_filepath)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Wikipedia parallel titles to page IDs"
    )
    parser.add_argument(
        "input_file", help="Path to the input file containing parallel titles"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "de", "hi", "it", "ko"],
        help="Language codes in order (default: en de hi it ko)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    # Extract languages from the first line if not provided
    if args.languages == ["en", "de", "hi", "it", "ko"]:
        with open(args.input_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if " ||| " in first_line:
                detected_languages = [
                    lang.strip() for lang in first_line.split(" ||| ")
                ]
                print(f"Detected languages from file: {detected_languages}")
                args.languages = detected_languages

    converter = ParallelTitleToPageIDConverter(args.languages)
    output_file = converter.convert_file(args.input_file)
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    main()
