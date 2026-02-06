#!/usr/bin/env python
"""
Tokenization Script for fair_sentence_transformers

This script tokenizes documents from various sources (jsonl, csv, txt, arrow)
and saves the tokenized dataset and metadata for later use.

Usage:
    python tokenize_dataset.py --config PATH_TO_CONFIG_FILE
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any
from datasets import DatasetDict

# Add project root to path to ensure imports work
project_root = str(Path(__file__).parent.parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fair_sentence_transformers.core.document_handler import DocumentHandler
from datasets import load_from_disk


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


def save_output_config(config: Dict[str, Any], output_path: str):
    """
    Save output configuration to a JSON file.

    Args:
        config: Configuration dictionary to save
        output_path: Path to save the configuration
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)


def tokenize_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process the dataset according to the configuration.

    Args:
        config: Configuration dictionary with processing parameters

    Returns:
        Updated configuration with output paths
    """
    # Extract configuration parameters
    dataset_path = config["dataset_path"]
    dataset_format = config["dataset_format"].lower()
    model_name = config["model_name"]

    # Make model name directory-friendly
    model_name_dir_friendly = model_name.replace("/", "_")

    # Determine output path
    output_path = config.get("output_path")

    if not output_path:
        # Create output path based on dataset path
        dataset_path_obj = Path(dataset_path)
        if dataset_path_obj.is_dir():
            output_path = os.path.join(
                dataset_path, f"tokenized__{model_name_dir_friendly}"
            )
        else:
            dataset_dir = os.path.dirname(dataset_path)
            output_path = os.path.join(
                dataset_dir, f"tokenized__{model_name_dir_friendly}"
            )

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Initialize document handler with the specified model
    document_handler = DocumentHandler(tokenizer_name=model_name)

    # Process according to dataset format
    print(f"Loading documents from {dataset_path} with format {dataset_format}...")

    if dataset_format == "arrow":
        # Load dataset from disk in arrow format
        print(f"Loading Hugging Face dataset from {dataset_path}...")
        dataset = load_from_disk(dataset_path)
        id_col = config.get("id_column", "id")
        text_col = config.get("text_column", "text")

        # Directly tokenize the loaded dataset
        print("Tokenizing arrow dataset...")
        tokenized_dataset = document_handler.tokenize_loaded_dataset(
            dataset=dataset, id_col=id_col, text_col=text_col
        )
    else:
        # Load documents based on the format
        if dataset_format == "jsonl":
            id_col = config.get("id_column", "id")
            text_col = config.get("text_column", "text")
            documents = document_handler.load_from_json_lines(
                filepath=dataset_path, id_col=id_col, text_col=text_col
            )
        elif dataset_format == "csv":
            id_col = config.get("id_column", "id")
            text_col = config.get("text_column", "text")
            documents = document_handler.load_from_csv(
                filepath=dataset_path, id_col=id_col, text_col=text_col
            )
        elif dataset_format == "txt":
            # Assuming the path is a directory containing text files
            documents = document_handler.load_from_text_files(directory=dataset_path)
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_format}")

        # Print information about loaded documents
        print(f"Loaded {len(documents)} documents")

        # Create tokenized dataset
        print("Tokenizing documents...")
        tokenized_dataset = document_handler.create_tokenized_dataset(documents)

    # Save tokenized dataset
    print(f"Saving tokenized dataset to {output_path}...")
    tokenized_dataset.save_to_disk(output_path)

    # Create and store metadata
    print("Creating and storing metadata...")
    if isinstance(tokenized_dataset, DatasetDict):
        # If it's a DatasetDict, we need to handle each split
        metadata = {}
        metadata_path = {}
        for split, dataset in tokenized_dataset.items():
            split_output_path = os.path.join(output_path, split)
            os.makedirs(split_output_path, exist_ok=True)
            split_metadata, split_metadata_path = (
                document_handler.create_and_store_metadata(
                    tokenized_dataset=dataset, directory_path=split_output_path
                )
            )
            metadata[split] = split_metadata
            metadata_path[split] = split_metadata_path
    else:
        # If it's a single Dataset, handle it directly
        metadata, metadata_path = document_handler.create_and_store_metadata(
            tokenized_dataset=tokenized_dataset, directory_path=output_path
        )

    # Update config with output information
    config["tokenized_dataset_path"] = output_path
    config["metadata_path"] = metadata_path

    # Save output config in the same directory as the tokenized dataset
    output_config_path = os.path.join(output_path, "tokenization_config.json")
    save_output_config(config, output_config_path)

    print(f"Tokenization complete. Output configuration saved to {output_config_path}")

    return config


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Tokenize documents for fair_sentence_transformers")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Process dataset
    tokenize_dataset(config)


if __name__ == "__main__":
    main()
