from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


@dataclass
class CustomDataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.0 (Volta).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padding_features = [
            {
                key: val
                for key, val in feature.items()
                if key in ["input_ids", "attention_mask", "token_type_ids"]
            }
            for feature in features
        ]

        batch = self.pad_without_fast_tokenizer_warning(
            self.tokenizer,
            padding_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        for feature in features:
            for key, value in feature.items():
                if key in ["input_ids", "attention_mask", "token_type_ids"]:
                    continue
                if key not in batch:
                    batch[key] = []
                batch[key].append(value)

        try:
            batch["length"] = torch.stack(batch["length"])
        except:
            pass

        return batch

    def pad_without_fast_tokenizer_warning(self, tokenizer, *pad_args, **pad_kwargs):
        """
        Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
        """

        # To avoid errors when using Feature extractors
        if not hasattr(tokenizer, "deprecation_warnings"):
            return tokenizer.pad(*pad_args, **pad_kwargs)

        # Save the state of the warning, then disable it
        warning_state = tokenizer.deprecation_warnings.get(
            "Asking-to-pad-a-fast-tokenizer", False
        )
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        try:
            padded = tokenizer.pad(*pad_args, **pad_kwargs)
        finally:
            # Restore the state of the warning.
            tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = (
                warning_state
            )

        return padded
