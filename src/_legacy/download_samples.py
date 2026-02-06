#!/usr/bin/env python
"""
Dataset Sampling Script for fair_sentence_transformers

This script downloads a specified number of random samples from a Hugging Face dataset
using streaming mode. It allows sampling without downloading the entire dataset.

Usage:
    python download_samples.py --dataset_url DATASET_URL --num_samples NUM_SAMPLES --output_path OUTPUT_PATH
    python download_samples.py --config PATH_TO_CONFIG_FILE
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict
import datetime

# Add project root to path to ensure imports work
project_root = str(Path(__file__).parent.parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets import load_dataset
from datasets import Dataset as RegularDataset
from fair_sentence_transformers.utils.wikipedia_id_translator import WikipediaCuridTranslator


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


def get_cohere_samples(
    dataset_url: str,
    num_samples: int,
    split: str = "train",
    source_lang: Optional[str] = None,  # eg "en"
    target_lang: Optional[str] = None,  # eg "it"
    source_batch_size: int = 10000,
    max_indices_searched_target: int = 100000,
) -> Tuple[RegularDataset, Dict[int, int]]:

    if not source_lang:
        raise ValueError("source_lang must be provided for Cohere dataset format")

    if target_lang:
        curid_translator = WikipediaCuridTranslator(source_lang, target_lang)
        source_data = load_dataset(
            dataset_url,
            source_lang,
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
        target_data = load_dataset(
            dataset_url,
            target_lang,
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
        source_para0 = source_data.filter(lambda row: row["paragraph_id"] == 0)

        source_para0 = source_para0.batch(batch_size=source_batch_size)
        matches_source2target_ids = {}
        print("Starting Source to Target ID search...")
        source_max_index = 0
        target_max_index = 0
        for i, batch in enumerate(source_para0):
            source_batch_no = i + 1
            print(f"Processing source batch {source_batch_no}...")
            source_wiki_ids = batch["wiki_id"]
            source_wiki_ids = [int(id) for id in source_wiki_ids if id is not None]
            source_max_index = max(
                source_max_index, max([int(id) for id in batch["id"]])
            )
            source2target_ids = curid_translator.translate_ids(source_wiki_ids)
            print("Source to Target IDs mapping completed.")
            target2source_ids = {
                int(v): int(k) for k, v in source2target_ids.items() if v is not None
            }
            target_para0 = target_data.filter(lambda row: (row["paragraph_id"] == 0))

            indices_searched_target = 0
            for sample in target_para0:
                target_wiki_id = (
                    int(sample["wiki_id"]) if sample["wiki_id"] is not None else None
                )

                if (
                    target_wiki_id is None
                    or target_wiki_id in matches_source2target_ids.values()
                ):
                    continue
                indices_searched_target += 1

                if target_wiki_id in target2source_ids:
                    matches_source2target_ids[target2source_ids[target_wiki_id]] = (
                        target_wiki_id
                    )
                    target_max_index = max(target_max_index, int(sample["id"]))
                if (
                    len(matches_source2target_ids) > num_samples
                ):  # last sample is used later to stop loop and cannot be used
                    break

                if indices_searched_target >= max_indices_searched_target:
                    sample_id = sample["id"]
                    print(
                        f"Reached max indices searched in target ({max_indices_searched_target}), stopping search for current source batch. Searched until id {sample_id}. Current size of matches: {len(matches_source2target_ids)}. Current datetime: {datetime.datetime.now()}"
                    )
                    break

            if len(matches_source2target_ids) > num_samples:
                assert len(matches_source2target_ids.keys()) == len(
                    set(matches_source2target_ids.values())
                )
                break

        print(f"Found {len(matches_source2target_ids)} matching source to target IDs.")
        print("Collecting source paragraphs for matched IDs...")
        source_wikiID2paras = defaultdict(list)
        # source_matches = source_data.filter(
        #     lambda row: row["wiki_id"] in matches_source2target_ids.keys()
        # )
        for i, sample in enumerate(source_data):
            if i > source_max_index:
                print(f"Reached the last source wiki ID: {source_max_index}, stopping.")
                break
            source_wiki_id = (
                int(sample["wiki_id"]) if sample["wiki_id"] is not None else None
            )
            if source_wiki_id not in matches_source2target_ids.keys():
                continue

            para_id = sample["paragraph_id"]
            text = sample["text"]
            source_wikiID2paras[source_wiki_id].insert(para_id, text)

        print("Source paragraphs collected.")

        print("Collecting target paragraphs for matched IDs...")
        target_wikiID2paras = defaultdict(list)
        # target_matches = target_data.filter(
        #     lambda row: row["wiki_id"] in matches_source2target_ids.values()
        # )
        for i, sample in enumerate(target_data):
            if i > target_max_index:
                print(f"Reached the last target wiki ID: {target_max_index}, stopping.")
                break
            target_wiki_id = (
                int(sample["wiki_id"]) if sample["wiki_id"] is not None else None
            )
            if target_wiki_id not in matches_source2target_ids.values():
                continue

            para_id = sample["paragraph_id"]
            text = sample["text"]
            target_wikiID2paras[target_wiki_id].insert(para_id, text)

        print("Target paragraphs collected.")

        print("Concatenating paragraphs...")
        source_wikiID2text = {
            wiki_id: "\n".join(paras) for wiki_id, paras in source_wikiID2paras.items()
        }
        target_wikiID2text = {
            wiki_id: "\n".join(paras) for wiki_id, paras in target_wikiID2paras.items()
        }

        print("Creating final dataset...")
        final_dataset_dict = {"id": [], "text": [], "language": [], "pair_id": []}
        for i, source_wikiID in zip(
            range(0, 2 * len(matches_source2target_ids), 2),
            matches_source2target_ids.keys(),
        ):
            target_wikiID = matches_source2target_ids[source_wikiID]
            final_dataset_dict["id"].append(i)
            final_dataset_dict["text"].append(source_wikiID2text[source_wikiID])
            final_dataset_dict["language"].append(source_lang)
            final_dataset_dict["pair_id"].append(i + 1)
            final_dataset_dict["id"].append(i + 1)
            final_dataset_dict["text"].append(target_wikiID2text[target_wikiID])
            final_dataset_dict["language"].append(target_lang)
            final_dataset_dict["pair_id"].append(i)

        final_dataset = RegularDataset.from_dict(final_dataset_dict)

        return final_dataset, matches_source2target_ids


def download_dataset_samples(
    dataset_url: str,
    num_samples: int,
    output_path: str,
    split: str = "train",
    shuffle: bool = True,
    seed: Optional[int] = 42,
    buffer_size: Optional[int] = 500000,
    output_format: str = "jsonl",
    dataset_format: Optional[str] = None,
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Download random samples from a Hugging Face dataset using streaming.

    Args:
        dataset_url: URL or name of the dataset on Hugging Face
        num_samples: Number of samples to download
        output_path: Path to save the downloaded samples
        split: Dataset split to use (default: "train")
        shuffle: Whether to shuffle the dataset (default: True)
        seed: Random seed for shuffling (default: 42)
        buffer_size: Size of shuffle buffer (default: 500000)
        output_format: Format to save samples in (default: "jsonl")
        dataset_format: Format of the dataset (optional, options include "cohere")

    Returns:
        Updated configuration with output information
    """
    if dataset_format and dataset_format.lower() == "cohere":
        print("Cohere dataset format detected, using custom loading logic...")
        samples, matches_source2target_ids = get_cohere_samples(
            dataset_url, num_samples, split, source_lang, target_lang
        )

    else:
        print(f"Loading dataset {dataset_url} in streaming mode...")
        # Load dataset in streaming mode
        dataset = load_dataset(
            dataset_url, split=split, streaming=True, trust_remote_code=True
        )

        # Shuffle the dataset if requested
        if shuffle:
            print(
                f"Shuffling dataset with seed {seed} and buffer size {buffer_size}..."
            )
            dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)

        # Prepare to extract samples
        print(f"Generating {num_samples} samples from dataset...")
        samples = dataset.take(num_samples)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Downloading samples and saving to {output_path}...")
    # Save the samples based on the specified format
    if output_format.lower() == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
    elif output_format.lower() == "arrow":
        # Convert to regular dataset and save as Arrow format
        sample_list = list(samples)
        regular_dataset = RegularDataset.from_dict(
            {k: [s[k] for s in sample_list] for k in sample_list[0].keys()}
        )
        regular_dataset.save_to_disk(output_path)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    # Create result config
    result_config = {
        "dataset_url": dataset_url,
        "num_samples": num_samples,
        "output_path": output_path,
        "split": split,
        "shuffle": shuffle,
        "seed": seed,
        "output_format": output_format,
        "dataset_format": dataset_format,
        "source_lang": source_lang,
        "target_lang": target_lang,
    }

    print(f"Successfully saved {num_samples} samples to {output_path}")

    return result_config


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Download samples from a Hugging Face dataset"
    )

    # Add arguments for either direct parameter passing or config file
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="Path to configuration file")
    group.add_argument(
        "--dataset_url", help="URL or name of the dataset on Hugging Face"
    )

    # Add optional arguments
    parser.add_argument("--num_samples", type=int, help="Number of samples to download")
    parser.add_argument("--output_path", help="Path to save the downloaded samples")
    parser.add_argument(
        "--split", default="train", help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=True,
        help="Whether to shuffle the dataset (default: True)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=500000,
        help="Size of shuffle buffer (default: 500000)",
    )
    parser.add_argument(
        "--output_format",
        default="jsonl",
        help="Format to save samples in (default: jsonl)",
    )
    parser.add_argument(
        "--dataset_format",
        choices=["cohere"],
        help="Format of the dataset (optional, options include 'cohere')",
    )
    parser.add_argument(
        "--source_lang",
        help="Source language for Cohere dataset format (e.g., 'en')",
    )
    parser.add_argument(
        "--target_lang",
        help="Target language for Cohere dataset format (e.g., 'it')",
    )

    args = parser.parse_args()

    # Process based on whether config file or direct parameters were provided
    if args.config:
        # Load configuration from file
        config = load_config(args.config)

        # Extract required parameters
        dataset_url = config["dataset_url"]
        num_samples = config["num_samples"]
        output_path = config["output_path"]

        # Extract optional parameters with defaults
        split = config.get("split", "train")
        shuffle = config.get("shuffle", True)
        seed = config.get("seed", 42)
        buffer_size = config.get("buffer_size", 500000)
        output_format = config.get("output_format", "jsonl")
        dataset_format = config.get("dataset_format")
        source_lang = config.get("source_lang")
        target_lang = config.get("target_lang")
    else:
        # Ensure required parameters are provided when not using config
        if not args.num_samples:
            parser.error("--num_samples is required when not using a config file")
        if not args.output_path:
            parser.error("--output_path is required when not using a config file")

        # Use command-line arguments
        dataset_url = args.dataset_url
        num_samples = args.num_samples
        output_path = args.output_path
        split = args.split
        shuffle = args.shuffle
        seed = args.seed
        buffer_size = args.buffer_size
        output_format = args.output_format
        dataset_format = args.dataset_format
        source_lang = args.source_lang
        target_lang = args.target_lang

    # Download dataset samples
    result_config = download_dataset_samples(
        dataset_url=dataset_url,
        num_samples=num_samples,
        output_path=output_path,
        split=split,
        shuffle=shuffle,
        seed=seed,
        buffer_size=buffer_size,
        output_format=output_format,
        dataset_format=dataset_format,
        source_lang=source_lang,
        target_lang=target_lang,
    )

    # Save output configuration - check if output_path is a directory
    output_path_obj = Path(output_path)
    if output_path_obj.is_dir():
        # If output_path is a directory, save the config file inside it
        config_filename = f"{output_path_obj.stem}_config.json"
        output_config_path = os.path.join(output_path, config_filename)
    else:
        # If output_path is a file, save the config alongside it
        output_config_path = os.path.join(
            os.path.dirname(output_path), f"{output_path_obj.stem}_config.json"
        )

    save_output_config(result_config, output_config_path)
    print(f"Output configuration saved to {output_config_path}")


if __name__ == "__main__":
    main()
