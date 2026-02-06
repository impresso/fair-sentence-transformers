"""
Embedding I/O Module

This module provides utilities for saving and loading embedding dictionaries to/from disk.
It supports both standalone embeddings and late-chunking embeddings with their specific formats.
"""

import os
import json
import torch
from typing import Dict, Optional, Tuple


def save_embeddings(
    embeddings: Dict[str, torch.Tensor],
    output_dir: str,
    filename: str = "embeddings",
    create_dir: bool = True,
) -> str:
    """
    Save embeddings dictionary to disk.

    Args:
        embeddings: Dictionary mapping IDs to embeddings (PyTorch tensors)
        output_dir: Directory to save the embeddings in
        filename: Base filename without extension
        create_dir: Whether to create the output directory if it doesn't exist

    Returns:
        Path to the saved embeddings file
    """
    # Create output directory if needed
    if create_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Define file path
    tensor_path = os.path.join(output_dir, f"{filename}.pt")

    # Convert embeddings dictionary to a serializable format
    # We'll separate keys and tensors
    keys = list(embeddings.keys())

    # Stack embeddings as a 2D tensor if they all have the same shape
    first_tensor = next(iter(embeddings.values()))
    tensor_shape = first_tensor.shape
    all_same_shape = all(tensor.shape == tensor_shape for tensor in embeddings.values())

    if all_same_shape:
        # Stack all tensors into a single tensor
        tensor_values = torch.stack(list(embeddings.values()))
        torch.save({"keys": keys, "tensors": tensor_values}, tensor_path)
    else:
        # Save as dictionary mapping keys to individual tensors
        torch.save(
            {"keys": keys, "tensors": {k: embeddings[k] for k in keys}}, tensor_path
        )

    return tensor_path


def load_embeddings(
    directory: str,
    filename: str = "embeddings",
    device: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Load embeddings dictionary from disk.

    Args:
        directory: Directory containing the embeddings
        filename: Base filename without extension
        device: Device to load tensors to ('cpu', 'cuda', 'mps').
               If None, will use tensors' current device.

    Returns:
        Dictionary mapping IDs to embeddings (PyTorch tensors)
    """
    # Define file path
    tensor_path = os.path.join(directory, f"{filename}.pt")

    # Load tensors
    data = torch.load(tensor_path, map_location=device if device else "cpu")
    keys = data["keys"]

    # Reconstruct embeddings dictionary
    embeddings = {}

    if isinstance(data["tensors"], torch.Tensor):  # Stacked tensors
        for i, key in enumerate(keys):
            embeddings[key] = data["tensors"][i]
    else:  # Dictionary of tensors
        for key in keys:
            embeddings[key] = data["tensors"][key]

    # print(f"Loaded {len(embeddings)} embeddings from {tensor_path}")
    return embeddings


def save_standalone_embeddings(
    embeddings: Dict[str, torch.Tensor],
    output_dir: str,
    pooling_strategy: str = "mean",
    model_name: Optional[str] = None,
) -> str:
    """
    Save standalone embeddings.

    Args:
        embeddings: Dictionary mapping document IDs to embeddings
        output_dir: Directory to save embeddings in
        pooling_strategy: Pooling strategy used (e.g., 'mean', 'cls')
        model_name: Optional name of the model used to generate embeddings

    Returns:
        Path to the saved embeddings file
    """
    # Construct filename based on whether model_name is provided
    if model_name:
        filename = f"standalone__{pooling_strategy}__{model_name.replace('/', '_')}"
    else:
        filename = f"standalone__{pooling_strategy}"

    return save_embeddings(
        embeddings=embeddings,
        output_dir=output_dir,
        filename=filename,
    )


def save_latechunking_embeddings(
    embeddings: Dict[str, torch.Tensor],
    output_dir: str,
    embedding_type: str = "segments",
    model_name: Optional[str] = None,
) -> str:
    """
    Save late-chunking embeddings.

    Args:
        embeddings: Dictionary mapping complex IDs to embeddings
        output_dir: Directory to save embeddings in
        embedding_type: Type of late-chunking embedding (e.g., 'segments', 'cls')
        model_name: Optional name of the model used to generate embeddings

    Returns:
        Path to the saved embeddings file
    """
    # Construct filename based on whether model_name is provided
    if model_name:
        filename = f"latechunking__{embedding_type}__{model_name.replace('/', '_')}"
    else:
        filename = f"latechunking__{embedding_type}"

    return save_embeddings(
        embeddings=embeddings,
        output_dir=output_dir,
        filename=filename,
    )


def load_standalone_embeddings(
    directory: str,
    pooling_strategy: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Load standalone embeddings from disk.

    Args:
        directory: Directory containing embeddings
        model_name: Name of model (if None, will attempt to infer from filenames)
        pooling_strategy: Pooling strategy (e.g., 'mean', 'cls')
        device: Device to load tensors to

    Returns:
        Dictionary mapping IDs to embeddings (PyTorch tensors).
        Format:
        {
            'segmentID1': tensor1,
            '3461318': tensor2,
            ...
        }
    """
    # First, check if there's an embedding_config.json file that defines the paths
    config_path = os.path.join(directory, "embedding_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

        if "embedding_paths" in config:
            # Use the specific path from the config based on pooling strategy
            if (
                pooling_strategy == "mean"
                and "standalone_mean" in config["embedding_paths"]
            ):
                embedding_path = config["embedding_paths"]["standalone_mean"]
                return load_embeddings(
                    os.path.dirname(embedding_path),
                    os.path.basename(embedding_path).replace(".pt", ""),
                    device,
                )
            elif (
                pooling_strategy == "cls"
                and "standalone_cls" in config["embedding_paths"]
            ):
                embedding_path = config["embedding_paths"]["standalone_cls"]
                return load_embeddings(
                    os.path.dirname(embedding_path),
                    os.path.basename(embedding_path).replace(".pt", ""),
                    device,
                )

    # If no config file with paths or specified path not found, fall back to searching for files
    files = os.listdir(directory)

    # Define patterns to look for based on pooling strategy
    if pooling_strategy == "mean":
        patterns = ["standalone__mean", "standalone_mean", "mean_standalone"]
    elif pooling_strategy == "cls":
        patterns = ["standalone__cls", "standalone_cls", "cls_standalone"]
    else:
        patterns = ["standalone__", "standalone_"]

    # Filter files by patterns
    matching_files = []
    for pattern in patterns:
        matching_files.extend([f for f in files if pattern in f and f.endswith(".pt")])

    if not matching_files:
        raise FileNotFoundError(
            f"No standalone {pooling_strategy if pooling_strategy else ''} embedding files found in {directory}"
        )

    # Use the first matching file
    filename = os.path.splitext(matching_files[0])[0]

    return load_embeddings(directory, filename, device)


def load_latechunking_embeddings(
    directory: str,
    model_name: Optional[str] = None,
    embedding_type: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Load late chunking embeddings from disk.

    Args:
        directory: Directory containing embeddings
        model_name: Name of model (if None, will attempt to infer from filenames)
        embedding_type: Type of late-chunking embedding (e.g., 'segments', 'cls', 'mean')
        device: Device to load tensors to

    Returns:
        Dictionary mapping IDs to embeddings (PyTorch tensors).
        Format for segments:
        {
            'segmentID1__pos0__seq_segmentID1_segmentID2_segmentID3': tensor1,
            '3539177__pos0__seq_3539177_124217_233347': tensor2,
            ...
        }

        Format for CLS & MEAN:
        {
            'seq_segmentID1_segmentID2_segmentID3': tensor1,
            'seq_3539177_124217_233347': tensor2,
            ...
        }
    """
    # First, check if there's an embedding_config.json file that defines the paths
    config_path = os.path.join(directory, "embedding_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

        if "embedding_paths" in config:
            # Use the specific path from the config based on embedding type
            config_key = None
            if embedding_type == "segments":
                config_key = "latechunking_segments"
            elif embedding_type == "cls":
                config_key = "latechunking_cls"
            elif embedding_type == "mean":
                config_key = "latechunking_mean"

            if config_key and config_key in config["embedding_paths"]:
                embedding_path = config["embedding_paths"][config_key]
                return load_embeddings(
                    os.path.dirname(embedding_path),
                    os.path.basename(embedding_path).replace(".pt", ""),
                    device,
                )

    # If no config file with paths or specified path not found, fall back to searching for files
    files = os.listdir(directory)

    # Define patterns to look for based on embedding type
    if embedding_type == "segments":
        patterns = [
            "latechunking__segments",
            "latechunking_segments",
            "segments_latechunking",
        ]
    elif embedding_type == "cls":
        patterns = ["latechunking__cls", "latechunking_cls", "cls_latechunking"]
    elif embedding_type == "mean":
        patterns = ["latechunking__mean", "latechunking_mean", "mean_latechunking"]
    else:
        patterns = ["latechunking__", "latechunking_"]

    # Filter files by patterns
    matching_files = []
    for pattern in patterns:
        matching_files.extend([f for f in files if pattern in f and f.endswith(".pt")])

    if not matching_files:
        raise FileNotFoundError(
            f"No late chunking {embedding_type if embedding_type else ''} embedding files found in {directory}"
        )

    # Use the first matching file
    filename = os.path.splitext(matching_files[0])[0]

    return load_embeddings(directory, filename, device)
