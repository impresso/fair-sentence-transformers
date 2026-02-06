"""
Document Handler Module

This module provides functionality for loading, preprocessing, and managing documents
for fair_sentence_transformers operations. It handles document loading from
various sources, tokenization, and preparation of documents for embedding models.
"""

import os
from typing import List, Dict, Tuple, Union
import polars as pl
from transformers import AutoTokenizer
from fair_sentence_transformers.utils.custom_data_collator import CustomDataCollatorWithPadding
from datasets import Dataset, DatasetDict
import torch
from torch.utils.data import DataLoader


class DocumentHandler:
    """
    Class to load, preprocess and manage documents for embedding operations.

    This class provides utilities to:
    1. Load documents from text files or CSV files
    2. Tokenize documents using HuggingFace tokenizers
    3. Create PyTorch DataLoaders with proper padding for batch processing
    4. Prepare documents for contextualized embedding models
    """

    def __init__(self, tokenizer_name: str = "bert-base-uncased"):
        """
        Initialize the DocumentHandler with a HuggingFace tokenizer.

        Args:
            tokenizer_name: Name or path of the HuggingFace tokenizer to use.
                            Defaults to 'bert-base-uncased'.
        """
        if tokenizer_name in ("Qwen/Qwen3-Embedding-0.6B"):
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, padding_side="left"
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def load_from_text_files(self, directory: str) -> List[Dict[str, str]]:
        """
        Load documents from text files in a directory.

        The function scans the specified directory for .txt files, reads their
        content, and creates a list of documents. The document ID is derived
        from the filename (without extension).

        Args:
            directory: Path to directory containing text files

        Returns:
            List of document dictionaries with 'id' and 'text' keys
        """
        documents = []

        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                doc_id = os.path.splitext(filename)[0]
                documents.append({"id": doc_id, "text": text})

        return documents

    def load_from_csv(
        self, filepath: str, id_col: str = "id", text_col: str = "text"
    ) -> List[Dict[str, str]]:
        """
        Load documents from a CSV file.

        The function reads a CSV file using Polars and extracts document IDs and
        text content from specified columns.

        Args:
            filepath: Path to CSV file
            id_col: Name of the column containing document IDs. Defaults to 'id'.
            text_col: Name of the column containing document text. Defaults to 'text'.

        Returns:
            List of document dictionaries with 'id' and 'text' keys
        """
        df = pl.read_csv(filepath)
        documents = []

        for row in df.iter_rows(named=True):
            documents.append({"id": str(row[id_col]), "text": str(row[text_col])})

        return documents

    def load_from_json_lines(
        self, filepath: str, id_col: str = "id", text_col: str = "text"
    ) -> List[Dict[str, str]]:
        """
        Load documents from a JSON Lines file.

        The function reads a JSON Lines file (jsonl), where each line contains
        a valid JSON object, and extracts document IDs and text content from
        specified fields.

        Args:
            filepath: Path to JSON Lines file
            id_col: Name of the field containing document IDs. Defaults to 'id'.
            text_col: Name of the field containing document text. Defaults to 'text'.

        Returns:
            List of document dictionaries with 'id' and 'text' keys
        """
        import json

        documents = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    try:
                        data = json.loads(line)
                        if id_col in data and text_col in data:
                            documents.append(
                                {"id": str(data[id_col]), "text": str(data[text_col])}
                            )
                        else:
                            print(
                                f"Warning: Missing columns {id_col} or {text_col} in line: {line[:100]}..."
                            )
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e} in line: {line[:100]}...")

        return documents

    def create_tokenized_dataset(self, documents: List[Dict[str, str]]) -> Dataset:
        """
        Convert documents into a tokenized Hugging Face dataset.

        This method tokenizes text documents using the initialized tokenizer and
        creates a Hugging Face Dataset object. The tokenization is performed in
        batches for efficiency and includes token counting.

        Args:
            documents: List of document dictionaries with 'id' and 'text' keys

        Returns:
            A Hugging Face Dataset containing the tokenized documents with
            input_ids, lengths, and original document identifiers
        """
        # Convert list of dictionaries to a Hugging Face Dataset
        dataset = Dataset.from_dict(
            {
                "id": [str(doc["id"]) for doc in documents],
                "text": [doc["text"] for doc in documents],
            }
        )

        # Define tokenization function
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                padding=False,
                truncation=True,
                add_special_tokens=True,
                return_length=True,
            )
            return tokenized

        # Apply tokenization in parallel
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Set format for PyTorch
        tokenized_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "length"],
            output_all_columns=True,
        )

        return tokenized_dataset

    def tokenize_loaded_dataset(
        self, dataset: Dataset, id_col: str = "id", text_col: str = "text"
    ) -> Dataset:
        """
        Tokenize an already loaded Hugging Face dataset.

        This method is optimized for pre-loaded datasets (e.g., in arrow format) and
        directly tokenizes them without intermediate conversion to dictionaries.

        Args:
            dataset: A loaded Hugging Face Dataset with text content
            id_col: Name of the column containing document IDs. Defaults to 'id'.
            text_col: Name of the column containing document text. Defaults to 'text'.

        Returns:
            A Hugging Face Dataset containing the tokenized documents
        """
        # Rename columns if necessary to standardize
        if id_col != "id" or text_col != "text":
            dataset = dataset.rename_column(id_col, "id")
            dataset = dataset.rename_column(text_col, "text")

        # Define tokenization function
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                padding=False,
                truncation=True,
                add_special_tokens=True,
                return_length=True,
            )
            return tokenized

        # Apply tokenization in parallel
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Set format for PyTorch
        tokenized_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "length"],
            output_all_columns=True,
        )

        return tokenized_dataset

    def get_dataloader(
        self, dataset: Dataset, batch_size: int = 32, shuffle: bool = False
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader for batch processing of tokenized documents.

        This method creates a DataLoader with a custom collator that handles
        dynamic padding of sequences to the maximum length in each batch,
        optimizing memory usage and processing efficiency.

        Args:
            dataset: A Hugging Face Dataset containing tokenized documents
            batch_size: Number of samples per batch. Defaults to 32.
            shuffle: Whether to shuffle the dataset. Defaults to False.

        Returns:
            A PyTorch DataLoader configured for efficient batch processing
        """
        data_collator = CustomDataCollatorWithPadding(self.tokenizer)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator
        )
        return dataloader

    def concat_tokenized_documents(
        self, documents: List[Dict[str, any]], separator: str = " "
    ) -> Dict[str, any]:
        """
        Concatenate multiple tokenized documents with a specified separator.

        This method takes multiple tokenized documents (as returned by create_tokenized_dataset)
        and concatenates them using the specified separator. Special tokens like CLS and SEP
        from the original documents are handled appropriately (removed from the middle).

        Args:
            documents: List of tokenized documents, each containing 'input_ids', 'id', 'text', etc.
            separator: String separator to use between documents. Defaults to a space.

        Returns:
            A dictionary containing the concatenated document with:
            - input_ids: Tensor of token IDs for the concatenated document
            - token_type_ids: Tensor of token type IDs
            - attention_mask: Tensor of attention masks
            - length: Original lengths of each document
            - doc_boundaries: Tensor indicating start/end indices for each document (excluding special tokens); indices follow Python slicing convention (start inclusive, end exclusive)
            - doc_ids: List of original document IDs
            - doc_texts: List of original document texts
        """
        if len(documents) < 2:
            raise ValueError(
                "At least two documents must be provided for concatenation"
            )

        # Get tokenized separator if needed
        separator_ids = None
        if separator:
            separator_ids = self.tokenizer.encode(separator, add_special_tokens=False)

        # Extract information from documents
        doc_ids = [str(doc["id"]) for doc in documents]
        doc_texts = [doc["text"] for doc in documents]
        input_ids_list = [doc["input_ids"] for doc in documents]
        lengths = [doc["length"] for doc in documents]

        # Remove special tokens (like CLS and SEP) from the documents except for the first CLS and last SEP
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        if self.tokenizer.name_or_path in ("Qwen/Qwen3-Embedding-0.6B"):
            sep_token_id = (
                self.tokenizer.pad_token_id
            )  # For Qwen3, pad token is used as separator

        # Process first document - keep CLS but remove SEP
        first_doc_ids = input_ids_list[0]
        if first_doc_ids[-1] == sep_token_id:
            first_doc_clean = first_doc_ids[:-1]
        else:
            first_doc_clean = first_doc_ids

        # Process middle documents - remove both CLS and SEP
        middle_docs_clean = []
        for i in range(1, len(input_ids_list) - 1):
            doc_ids_i = input_ids_list[i]
            start_idx = 1 if doc_ids_i[0] == cls_token_id else 0
            end_idx = -1 if doc_ids_i[-1] == sep_token_id else len(doc_ids_i)
            middle_docs_clean.append(doc_ids_i[start_idx:end_idx])

        # Process last document - remove CLS but keep SEP
        last_doc_ids = input_ids_list[-1]
        if last_doc_ids[0] == cls_token_id:
            last_doc_clean = last_doc_ids[1:]
        else:
            last_doc_clean = last_doc_ids

        # Concatenate all documents with separator
        concatenated_ids = [first_doc_clean]
        for doc_clean in middle_docs_clean:
            if separator_ids:
                concatenated_ids.append(torch.tensor(separator_ids))
            concatenated_ids.append(doc_clean)

        if separator_ids and len(documents) > 1:
            concatenated_ids.append(torch.tensor(separator_ids))
        concatenated_ids.append(last_doc_clean)

        # Concatenate all tensors
        concatenated_input_ids = torch.cat(concatenated_ids)

        # Calculate document boundaries (excluding special tokens)
        doc_boundaries = []
        if cls_token_id:
            current_idx = 1  # Start after CLS token
        else:
            current_idx = 0

        for i, doc in enumerate(documents):
            doc_length = len(input_ids_list[i])

            # Adjust for special tokens
            if i == 0:  # First document
                doc_length = (
                    doc_length
                    - (1 if input_ids_list[i][-1] == sep_token_id else 0)
                    - (1 if input_ids_list[i][0] == cls_token_id else 0)
                )  # Subtract CLS
                doc_end = current_idx + doc_length
                doc_boundaries.append([current_idx, doc_end])
                current_idx = doc_end
            elif i == len(documents) - 1:  # Last document
                doc_length = (
                    doc_length
                    - (1 if input_ids_list[i][0] == cls_token_id else 0)
                    - (1 if input_ids_list[i][-1] == sep_token_id else 0)
                )
                if separator_ids:
                    current_idx += len(separator_ids)
                doc_end = current_idx + doc_length
                doc_boundaries.append([current_idx, doc_end])
            else:  # Middle documents
                doc_length = (
                    doc_length
                    - (1 if input_ids_list[i][0] == cls_token_id else 0)
                    - (1 if input_ids_list[i][-1] == sep_token_id else 0)
                )
                if separator_ids:
                    current_idx += len(separator_ids)
                doc_end = current_idx + doc_length
                doc_boundaries.append([current_idx, doc_end])
                current_idx = doc_end

        # Create token_type_ids (all 0s)
        token_type_ids = torch.zeros_like(concatenated_input_ids)

        # Create attention_mask (1s for all tokens)
        attention_mask = torch.ones_like(concatenated_input_ids)

        # Create final output
        concat_doc = {
            "id": f"seq_{'_'.join(doc_ids)}",
            "text": separator.join(doc_texts),
            "input_ids": concatenated_input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "length": torch.tensor(lengths),
            "doc_boundaries": torch.tensor(doc_boundaries),
            "doc_ids": doc_ids,
            "doc_texts": doc_texts,
        }

        return concat_doc

    def create_dataset_from_tokenized_docs_list(
        self, list_tokenized_docs: List[Dict[str, any]]
    ) -> Dataset:
        """
        Create a Hugging Face Dataset from a list of tokenized documents.

        This method takes a list of tokenized documents (e.g., a list of concatenated documents as returned by
        concat_tokenized_documents) and converts it into a Hugging Face Dataset.

        Args:
            tokenized_docs: List of tokenized documents, each containing 'input_ids', 'id', 'text', etc.

        Returns:
            A Hugging Face Dataset containing the tokenized documents
        """

        def generator(list_of_dicts):
            for document_dict in list_of_dicts:
                yield document_dict

        dataset = Dataset.from_generator(
            generator=generator,
            gen_kwargs={"list_of_dicts": list_tokenized_docs},
        )
        return dataset

    def create_concatenated_dataset(
        self,
        dataset: Dataset,
        concat_indices: List[List[int]],
        separator: str = " ",
    ) -> Dataset:
        """
        Create a dataset of concatenated documents based on provided indices.

        This function creates a new dataset by concatenating documents from the input dataset
        according to the indices specified in the concat_indices nested list. Each inner list
        represents one sample with the document indices to concatenate in the desired order.

        Args:
            dataset: A Hugging Face Dataset of tokenized documents (output of create_tokenized_dataset)
            concat_indices: A nested list where each inner list contains dataset indices to concatenate
            separator: String to use as separator between concatenated documents. Defaults to space.

        Returns:
            A new Hugging Face Dataset containing the concatenated documents

        Raises:
            ValueError: If any index in concat_indices is out of range for the dataset
        """
        # Convert dataset to list for easier indexing
        dataset_list = list(dataset)
        dataset_size = len(dataset_list)

        # Validate indices
        for i, indices in enumerate(concat_indices):
            if not indices:  # Empty list
                raise ValueError(f"Empty index list at position {i}")
            if len(indices) < 2:
                raise ValueError(
                    f"Each index list must contain at least 2 indices, but list at position {i} contains {len(indices)}"
                )
            for idx in indices:
                if idx < 0 or idx >= dataset_size:
                    raise ValueError(
                        f"Index {idx} in list at position {i} is out of range for dataset with size {dataset_size}"
                    )

        concatenated_docs = []

        # Create concatenated samples based on the provided indices
        for indices in concat_indices:
            # Get the documents corresponding to the indices
            selected_docs = [dataset_list[i] for i in indices]

            # Concatenate the selected documents
            concatenated_doc = self.concat_tokenized_documents(
                selected_docs, separator=separator
            )
            concatenated_docs.append(concatenated_doc)

        # Create a new dataset from the list of concatenated documents
        return self.create_dataset_from_tokenized_docs_list(concatenated_docs)

    def create_standalone_dataset(
        self,
        dataset: Dataset,
        indices: List[int],
    ) -> Dataset:
        """
        Create a new dataset by filtering the input dataset to contain only documents at specified indices.

        This function creates a new dataset by selecting documents from the input dataset
        according to the indices provided. Unlike create_concatenated_dataset, this method
        does not concatenate documents but simply filters them.

        Args:
            dataset: A Hugging Face Dataset of tokenized documents (output of create_tokenized_dataset)
            indices: A list of indices specifying which documents to include in the output dataset

        Returns:
            A new Hugging Face Dataset containing only the documents at the specified indices

        Raises:
            ValueError: If any index is out of range for the dataset
        """
        # Convert dataset to list for easier indexing
        dataset_list = list(dataset)
        dataset_size = len(dataset_list)

        # Validate indices
        for idx in indices:
            if idx < 0 or idx >= dataset_size:
                raise ValueError(
                    f"Index {idx} is out of range for dataset with size {dataset_size}"
                )

        # Filter the dataset to include only documents at specified indices
        filtered_docs = [dataset_list[idx] for idx in indices]

        # Create a new dataset from the filtered list of documents
        return self.create_dataset_from_tokenized_docs_list(filtered_docs)

    def prepare_datasets(
        self,
        dataset: Dataset,
        concat_indices: List[List[int]],
        standalone_indices: List[int],
        separator: str = " ",
    ) -> Dict[str, Dataset]:
        """
        Create both concatenated and standalone datasets in a single call.

        This convenience method creates two datasets:
        1. A concatenated dataset based on concat_indices where each list represents indices to concatenate
        2. A standalone dataset containing only the documents at the indices specified in standalone_indices

        Args:
            dataset: A Hugging Face Dataset of tokenized documents (output of create_tokenized_dataset)
            concat_indices: A nested list where each inner list contains dataset indices to concatenate
            standalone_indices: A list of indices specifying which documents to include in the standalone dataset
            separator: String to use as separator between concatenated documents. Defaults to space.

        Returns:
            A dictionary containing both datasets:
            - 'concatenated': The dataset with concatenated documents
            - 'standalone': The dataset with filtered standalone documents

        Raises:
            ValueError: If any index is out of range for the dataset
        """
        concatenated_dataset = self.create_concatenated_dataset(
            dataset=dataset, concat_indices=concat_indices, separator=separator
        )

        standalone_dataset = self.create_standalone_dataset(
            dataset=dataset, indices=standalone_indices
        )

        return {"concatenated": concatenated_dataset, "standalone": standalone_dataset}

    def prepare_datasets_wiki_parallel(
        self,
        dataset_dict: DatasetDict,
        concat_indices: List[List[int]],
        # standalone_indices: List[int],
        separator: str = " ",
        source_lang: str = "en",
        target_lang: str = None,
    ) -> Tuple[Dict[str, Dataset], List[List[int]], List[int]]:

        # Prepare source dataset
        source_dataset = dataset_dict[source_lang]
        joined_dataset = {}
        for i in range(len(source_dataset)):
            source_doc = source_dataset[i]
            joined_dataset[source_doc["id"]] = source_doc

        # Multilingual case: if target_lang is provided, add target dataset to joined_dataset
        if target_lang:
            target_dataset = dataset_dict[target_lang]
            for i in range(len(target_dataset)):
                target_doc = target_dataset[i]
                joined_dataset[target_doc["id"]] = target_doc

        concat_indices_lang_codes = []
        for indices in concat_indices:
            lang_code_indices = []
            for i, idx in enumerate(indices):
                if target_lang and (
                    i == 0
                ):  # If in multilingual mode (i.e., target_lang is provided), then First segment is always in target language
                    lang_code_indices.append(f"{idx}{target_lang}")
                else:
                    lang_code_indices.append(f"{idx}{source_lang}")
            concat_indices_lang_codes.append(lang_code_indices)

        # Flatten concat_indices_lang_codes and get unique values
        standalone_indices_lang_codes = sorted(
            list({item for sublist in concat_indices_lang_codes for item in sublist})
        )

        # Create standalone dataset
        filtered_docs = []
        for id in standalone_indices_lang_codes:
            filtered_docs.append(joined_dataset[id])
        standalone_dataset = self.create_dataset_from_tokenized_docs_list(filtered_docs)

        # Create concatenated dataset
        concatenated_docs = []
        # Create concatenated samples based on the provided indices
        for ids in concat_indices_lang_codes:
            # Get the documents corresponding to the indices
            selected_docs = []
            for id in ids:
                selected_docs.append(joined_dataset[id])
            # Concatenate the selected documents
            concatenated_doc = self.concat_tokenized_documents(
                selected_docs, separator=separator
            )
            concatenated_docs.append(concatenated_doc)
        concatenated_dataset = self.create_dataset_from_tokenized_docs_list(
            concatenated_docs
        )

        datasets = {
            "concatenated": concatenated_dataset,
            "standalone": standalone_dataset,
        }
        return datasets, concat_indices_lang_codes, standalone_indices_lang_codes

    def create_and_store_metadata(
        self,
        tokenized_dataset: Dataset,
        directory_path: str,
        languages_map: Dict[str, str] = None,
        topics_map: Dict[str, str] = None,
        filename: str = "metadata.json",
    ) -> Tuple[Dict[str, Dict[str, Union[int, str]]], str]:
        """
        Compute and store metadata for a tokenized dataset.

        This function computes metadata for each document in the tokenized dataset
        and stores it in a JSON file in the specified directory.

        Args:
            tokenized_dataset: A Hugging Face Dataset containing tokenized documents
            directory_path: Path to directory where the metadata file will be stored
            languages_map: Optional mapping from document ID to language (defaults to "en")
            topics_map: Optional mapping from document ID to topic (defaults to "_NA")
            filename: Name of the metadata file to create (defaults to "metadata.json")

        Returns:
            A dictionary containing the metadata for each document
        """
        import json
        import os

        # Initialize empty maps if not provided
        languages_map = languages_map or {}
        topics_map = topics_map or {}

        # Create metadata dictionary
        metadata = {}
        for idx, doc in enumerate(tokenized_dataset):
            doc_id = str(doc["id"])
            # Subtract 2 from length to exclude start and end tokens
            token_length = (
                doc["length"].item() - 2
                if isinstance(doc["length"], torch.Tensor)
                else doc["length"] - 2
            )

            # Use provided mappings if available, otherwise use defaults
            language = doc.get("language", languages_map.get(doc_id, "en"))
            topic = doc.get("topic", topics_map.get(doc_id, "_NA"))

            # Extract pair_id if available
            pair_id = doc.get("pair_id", "_NA")

            metadata[doc_id] = {
                "dataset_idx": idx,
                "token_length": token_length,
                "language": language,
                "topic": topic,
                "pair_id": pair_id,
            }

        # Ensure the directory exists
        os.makedirs(directory_path, exist_ok=True)

        # Write metadata to file
        metadata_path = os.path.join(directory_path, filename)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved to {metadata_path}")
        return metadata, metadata_path
