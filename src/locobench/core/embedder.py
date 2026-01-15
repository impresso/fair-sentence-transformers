# src/locobench/core/embedder.py

"""
Document Embedder Module

This module provides classes for embedding documents using various strategies.
It supports both standalone embedding (where each document is embedded individually)
and late-chunking embedding (where documents are embedded as part of a larger context).
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import List, Dict, Union, Literal, Optional, Any, Tuple, Callable, Set
import tqdm
import importlib.util
from torch import Tensor
import nnsight
from types import SimpleNamespace
import numpy as np
import warnings
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

ATTN_CALIB_SUPPORTED_MODELS = ["Alibaba-NLP/gte-multilingual-base"]


class BaseEmbedder:
    """
    Base class for document embedders.

    Handles common functionality like model initialization and forward pass.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        normalize: bool = True,
        jina_v3_task: Optional[str] = "text-matching",
        apply_attn_calibration: bool = False,
        calib_layers: Optional[str] = None,
        calib_source_tokens: Optional[str] = None,
        calib_basket_size: Optional[int] = None,
        calib_strength: Optional[float] = 1.0,
        device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
    ):
        """
        Initialize the BaseEmbedder with a HuggingFace model.

        Args:
            model_name: Name or path of the HuggingFace model to use for embedding
            device: Device to run the model on ('cpu', 'cuda', 'mps').
                   If None, will use CUDA or MPS if available, else CPU.
            normalize: Whether to normalize the embeddings to unit length (L2 norm)
            jina_v3_task: Task type for Jina v3 embeddings model
        """
        if device is None:
            self.device = DEVICE
        else:
            self.device = device
        # Optional device_map for HF accelerate-style sharding across multiple GPUs
        self.device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = device_map

        # Check if xformers is available for memory-efficient attention
        spec = importlib.util.find_spec("xformers")
        if spec is None:
            print("Warning: xformers not found.")
            xformers_available = False
        else:
            xformers_available = True

        if apply_attn_calibration:
            assert (
                model_name in ATTN_CALIB_SUPPORTED_MODELS
            ), f"Attention calibration is only supported for the following models: {ATTN_CALIB_SUPPORTED_MODELS}"
            print("Loading model using eager attn to support calibration")
            # If device_map is provided, let HF shard the model across GPUs and do NOT move to a single device.
            if self.device_map is not None:
                hf_model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    attn_implementation="eager",
                    torch_dtype=torch.float16,
                    device_map=self.device_map,
                )
            else:
                hf_model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    attn_implementation="eager",
                    torch_dtype=torch.float16,
                ).to(self.device)
            self.model = nnsight.NNsight(hf_model)

            self.num_layers = int(self.model.config.num_hidden_layers)

            if calib_layers is None:
                raise ValueError(
                    "Must specify calib_layers when using attention calibration"
                )
            elif calib_layers == "last_half":
                self.calib_layer_idx_start = self.num_layers // 2
                self.calib_layer_idx_end = self.num_layers
            elif calib_layers == "last":
                self.calib_layer_idx_start = self.num_layers - 1
                self.calib_layer_idx_end = self.num_layers
            else:
                raise ValueError("Invalid calibration layer selection")

            if calib_source_tokens is None:
                raise ValueError(
                    "Must specify calib_source_tokens when using attention calibration"
                )
            elif calib_source_tokens == "cls":
                self.calib_source_mode = "cls"
                self.calib_source_token_idx = 0
            elif calib_source_tokens == "all":
                self.calib_source_mode = "all"
            else:
                raise ValueError("Invalid calibration source token selection")

            if calib_basket_size is None:
                raise ValueError(
                    "Must specify calib_basket_size when using attention calibration"
                )
            self.calib_basket_size = calib_basket_size
            assert calib_strength is not None
            assert isinstance(
                calib_strength, (float, int)
            ), f"calib_strength must be float-like, got {type(calib_strength)}"
            assert (
                0.0 <= float(calib_strength) <= 1.0
            ), "calib_strength must be in [0, 1]"
            self.calib_strength = float(calib_strength)
            # Track calibration stats across the run
            self.calib_short_seq_count: int = 0
            self.calib_total_seq_count: int = 0

        elif model_name == "Alibaba-NLP/gte-multilingual-base" and xformers_available:
            print("Loading Alibaba-NLP/gte-multilingual-base with xformers support")
            if self.device_map is not None:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    unpad_inputs=True,
                    use_memory_efficient_attention=True,
                    torch_dtype=torch.float16,
                    device_map=self.device_map,
                )
            else:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    unpad_inputs=True,
                    use_memory_efficient_attention=True,
                    torch_dtype=torch.float16,
                ).to(self.device)
        elif model_name == "Qwen/Qwen3-Embedding-0.6B" and xformers_available:
            if self.device_map is not None:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.float16,
                    device_map=self.device_map,
                )
            else:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.float16,
                    device_map=self.device,
                ).to(self.device)
        else:
            if self.device_map is not None:
                self.model = AutoModel.from_pretrained(
                    model_name, trust_remote_code=True, device_map=self.device_map
                )
            else:
                self.model = AutoModel.from_pretrained(
                    model_name, trust_remote_code=True
                ).to(self.device)

        self.model.eval()
        self.normalize = normalize
        self.model_name = model_name
        self.apply_attn_calibration = apply_attn_calibration

        # Set up Jina v3 task ID if applicable
        if model_name == "jinaai/jina-embeddings-v3":
            self.task_id = self.model._adaptation_map[jina_v3_task]

    def get_output_calibrated(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Any:
        # NOTE: Calibrate either CLS-only or all query rows depending on self.calib_source_mode.
        # Record stats about sequence lengths vs basket size before tracing (no side-effects inside trace).
        S_stat = int(self.calib_basket_size)
        valid_len_cpu = attention_mask.sum(dim=1)
        # Update counters: total sequences seen and how many are shorter than S
        self.calib_total_seq_count += int(valid_len_cpu.numel())
        self.calib_short_seq_count += int((valid_len_cpu < S_stat).sum().item())
        with self.model.trace() as tracer:
            with tracer.invoke(input_ids=input_ids, attention_mask=attention_mask):
                for i in range(self.calib_layer_idx_start, self.calib_layer_idx_end):
                    attn = self.model.encoder.layer[
                        i
                    ].attention.source.self__attention_0.source.nn_functional_softmax_0.output
                    orig_attn = attn.clone()
                    # Shape assertions (fail fast)
                    assert attn.dim() == 4, f"Expected attn to be 4D, got {attn.shape}"
                    B, H, Q, K = attn.shape
                    assert attention_mask.shape == (
                        B,
                        K,
                    ), f"attention_mask shape mismatch: {attention_mask.shape} vs expected {(B, K)}"
                    if getattr(self, "calib_source_mode", "cls") == "cls":
                        assert (
                            self.calib_source_token_idx < Q
                        ), f"CLS/source index {self.calib_source_token_idx} out of range for Q={Q}"
                    assert (
                        self.calib_basket_size is not None
                        and self.calib_basket_size > 0
                    ), "calib_basket_size must be a positive integer"

                    # Prepare mask (right-padding asserted)
                    mask = attention_mask.to(device=attn.device, dtype=torch.long)
                    valid_len = mask.sum(dim=1)  # (B,)
                    assert torch.all(valid_len > 0), "Empty sequences are not supported"
                    S = int(self.calib_basket_size)
                    pos = torch.arange(K, device=attn.device).unsqueeze(0)  # (1,K)
                    right_padded = (pos < valid_len.unsqueeze(1)).to(mask.dtype)
                    assert torch.equal(
                        mask, right_padded
                    ), "Attention mask must be right-padded (all valid tokens first)"

                    # For self-attention we assume Q == K.
                    assert (
                        Q == K
                    ), f"Expected self-attention with Q==K, got Q={Q}, K={K}"

                    # Select which query rows to calibrate: R = 1 for 'cls', else R = Q for 'all'
                    if getattr(self, "calib_source_mode", "cls") == "cls":
                        q_idx = torch.tensor(
                            [self.calib_source_token_idx], device=attn.device
                        )
                    else:
                        q_idx = torch.arange(Q, device=attn.device)
                    R = q_idx.numel()

                    # Extract selected query rows: (B, H, R, K)
                    sel_rows = attn.index_select(dim=2, index=q_idx)

                    # Zero out pads upfront using key mask
                    mask_f = right_padded.to(dtype=attn.dtype)  # (B,K)
                    mask_bk = mask_f.view(B, 1, 1, K)  # (B,1,1,K)
                    sel_rows_masked = sel_rows * mask_bk  # (B,H,R,K)

                    # Query validity mask (to avoid normalizing padded query rows)
                    qmask = right_padded[:, :Q].to(dtype=torch.bool)  # (B,Q)
                    qmask_sel = qmask.index_select(dim=1, index=q_idx)  # (B,R)
                    qmask_sel_b = qmask_sel.view(B, 1, R, 1)  # (B,1,R,1)

                    # Compute basket ids per position (shared across batch/heads)
                    # Place CLS (pos==0) in its own basket 0; content starts at basket 1 in size-S groups.
                    content_buckets = torch.where(
                        pos == 0, torch.zeros_like(pos), 1 + ((pos - 1) // S)
                    )  # (1,K)
                    max_content_baskets = 0 if K <= 1 else ((K - 1 + (S - 1)) // S)
                    max_baskets = 1 + max_content_baskets
                    basket_ids = content_buckets.clamp_max(max_baskets - 1)  # (1,K)
                    basket_ids = basket_ids.view(1, 1, 1, K).expand(
                        B, H, R, K
                    )  # (B,H,R,K)

                    # Sum attention per basket: (B,H,R,max_baskets)
                    bucket_sums = torch.zeros(
                        (B, H, R, max_baskets), device=attn.device, dtype=attn.dtype
                    )
                    bucket_sums = bucket_sums.scatter_add(
                        dim=-1, index=basket_ids, src=sel_rows_masked
                    )

                    # Gather per-position denominators: (B,H,R,K)
                    denom = bucket_sums.gather(dim=-1, index=basket_ids)

                    # Number of baskets per item with CLS isolated:
                    # n_baskets = 1 (CLS) + ceil(max(valid_len-1, 0) / S)
                    n_content = ((torch.clamp_min(valid_len - 1, 0) + (S - 1)) // S).to(
                        dtype=attn.dtype
                    )
                    n_baskets = (1 + n_content).view(B, 1, 1, 1)  # (B,1,1,1)

                    # Apply calibration formula per position (masked positions remain zero)
                    scaled = sel_rows_masked * S  # (B,H,R,K)
                    calibrated = torch.where(
                        denom > 0,
                        (scaled / denom) * (1.0 / n_baskets),
                        torch.zeros_like(scaled),
                    )

                    # Enforce valid distribution over valid tokens (sum to 1), keep pads at 0
                    calibrated = calibrated * mask_bk
                    row_sum = calibrated.sum(dim=-1, keepdim=True)  # (B,H, R, 1)
                    # Only valid query rows must have positive sums
                    row_sum_squeezed = row_sum.squeeze(-1)  # (B,H,R)
                    must_be_pos = qmask_sel.view(B, 1, R).expand(B, H, R)
                    assert torch.all(
                        row_sum_squeezed[must_be_pos] > 0
                    ), "Calibrated row sums must be > 0 for valid queries"
                    # Normalize valid query rows, keep invalid query rows unchanged (original masked rows)
                    denom_rows = torch.where(
                        qmask_sel_b > 0, row_sum, torch.ones_like(row_sum)
                    )
                    normalized = torch.where(
                        qmask_sel_b > 0,
                        calibrated / denom_rows,
                        sel_rows_masked,
                    )

                    # Replace selected query rows in-place to avoid extra full-tensor allocation
                    attn.index_copy_(2, q_idx, normalized)

                    updated_attention = (
                        self.combine_recalibrated_and_original_attention(
                            attn, orig_attn, self.calib_strength
                        )
                    )
                    self.model.encoder.layer[
                        i
                    ].attention.source.self__attention_0.source.nn_functional_softmax_0.output = (
                        updated_attention
                    )

                # Request the final model output so it is materialized after the trace.
                # We expect to access last_hidden_state downstream.
                model_output = self.model.output.save()

        # After exiting the trace context, saved proxies are concrete values.
        # Ensure downstream code can access `.last_hidden_state` as an attribute.
        if hasattr(model_output, "last_hidden_state"):
            return model_output
        # Some HF models return dict-like outputs through NNsight; assert the key exists and wrap.
        assert isinstance(
            model_output, (dict, tuple, list)
        ), "Unexpected model output type from NNsight"
        if isinstance(model_output, dict):
            assert (
                "last_hidden_state" in model_output
            ), "Model output missing 'last_hidden_state'"
            return SimpleNamespace(last_hidden_state=model_output["last_hidden_state"])
        # Fallback for tuple/list where index 0 is last_hidden_state (common HF convention)
        assert len(model_output) > 0, "Empty model output sequence"
        return SimpleNamespace(last_hidden_state=model_output[0])

    def combine_recalibrated_and_original_attention(
        self,
        recalibrated_attention: torch.Tensor,
        original_attention: torch.Tensor,
        recalibration_strength: Union[float, torch.Tensor],
    ):
        """
        Blends recalibrated and original attention using linear interpolation.

        Args:
            recalibrated_attention: Tensor of recalibrated attention weights
            original_attention: Tensor of original attention weights (same shape as recalibrated)
            recalibration_strength: Float in [0, 1] controlling the blend ratio
                - 0.0: returns original attention only
                - 0.5: returns equal mix of both
                - 1.0: returns recalibrated attention only

        Returns:
            Blended attention tensor with same shape as inputs
        """
        assert (
            recalibrated_attention.shape == original_attention.shape
        ), "recalibrated_attention and original_attention must have the same shape"
        strength = torch.as_tensor(
            recalibration_strength,
            device=recalibrated_attention.device,
            dtype=recalibrated_attention.dtype,
        )
        assert strength.ndim == 0, "recalibration_strength must be a scalar"
        assert (
            0.0 <= float(strength.item()) <= 1.0
        ), "recalibration_strength must be in [0, 1]"
        return (recalibrated_attention * strength) + (
            original_attention * (1.0 - strength)
        )

    def get_model_outputs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Any:
        """
        Get model outputs with appropriate handling for special models.

        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask for the input

        Returns:
            Model outputs
        """
        # Move tensors to the correct device unless we're using a multi-GPU device_map.
        # With device_map, let HF accelerate handle dispatch; during calibration we move per-layer tensors as needed.
        if self.device_map is None:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

        # Apply attention calibration if enabled
        if self.apply_attn_calibration:
            with torch.no_grad():
                outputs = self.get_output_calibrated(
                    input_ids=input_ids, attention_mask=attention_mask
                )

        # If model is jinaai/jina-embeddings-v3, then add task_id
        elif self.model_name == "jinaai/jina-embeddings-v3":
            adapter_mask = torch.full(
                (input_ids.shape[0],),
                self.task_id,
                dtype=torch.int32,
                device=self.device,
            )
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    adapter_mask=adapter_mask,
                )
        # For other models, use the standard forward pass
        else:
            input_ids = input_ids.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return outputs

    def mean_pooling(
        self, last_hidden_state: torch.Tensor, input_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply mean pooling to the model output.

        Args:
            last_hidden_state: Hidden state output from the model
            input_attention_mask: Attention mask to consider only valid tokens

        Returns:
            Pooled embeddings
        """
        # Expand attention mask to match the shape of the model output
        attention_expanded = input_attention_mask.unsqueeze(-1).expand(
            last_hidden_state.size()
        )
        # Sum up the hidden states and divide by the actual sequence length
        sum_hidden_states = torch.sum(last_hidden_state * attention_expanded, dim=1)
        seq_lengths = torch.sum(input_attention_mask, dim=1, keepdim=True)
        return sum_hidden_states / seq_lengths  # (shape: [batch_size, hidden_size])

    def get_embeddings_from_hidden_states(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get both CLS and mean pooled embeddings from hidden states.

        Args:
            last_hidden_state: Hidden state output from the model
            attention_mask: Attention mask for the input

        Returns:
            Dictionary containing both types of embeddings ('cls' and 'mean')
        """
        if self.model_name in ("Qwen/Qwen3-Embedding-0.6B"):
            # For Qwen models, use the last token pooling strategy
            cls_embeddings = self.last_token_pool(last_hidden_state, attention_mask)
        else:
            # For other models, use the first token as CLS token
            cls_embeddings = last_hidden_state[:, 0]
            # (shape: [batch_size, hidden_size])

        # Get mean pooled embeddings
        mean_embeddings = self.mean_pooling(last_hidden_state, attention_mask)

        # Normalize if required
        if self.normalize:
            cls_embeddings = F.normalize(cls_embeddings, p=2, dim=1)
            mean_embeddings = F.normalize(mean_embeddings, p=2, dim=1)

        return {"cls": cls_embeddings, "mean": mean_embeddings}

    def last_token_pool(
        self, last_hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]


class StandaloneEmbedder(BaseEmbedder):
    """
    Class for creating embeddings from standalone documents.

    This class processes documents that have been individually tokenized
    and creates embeddings using a specified encoder model and pooling strategy.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        normalize: bool = True,
        jina_v3_task: Optional[str] = "text-matching",
        apply_attn_calibration: bool = False,
        calib_layers: Optional[str] = None,
        calib_source_tokens: Optional[str] = None,
        calib_basket_size: Optional[int] = None,
        calib_strength: Optional[float] = 1.0,
        device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
    ):
        """
        Initialize the StandaloneEmbedder with a HuggingFace model.

        Args:
            model_name: Name or path of the HuggingFace model to use for embedding
            device: Device to run the model on ('cpu', 'cuda', 'mps').
                   If None, will use CUDA or MPS if available, else CPU.
            normalize: Whether to normalize the embeddings to unit length (L2 norm)
            jina_v3_task: Task type for Jina v3 embeddings model
        """
        super().__init__(
            model_name=model_name,
            device=device,
            normalize=normalize,
            jina_v3_task=jina_v3_task,
            apply_attn_calibration=apply_attn_calibration,
            calib_layers=calib_layers,
            calib_source_tokens=calib_source_tokens,
            calib_basket_size=calib_basket_size,
            calib_strength=calib_strength,
            device_map=device_map,
        )

    def embed_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[List, torch.Tensor]]:
        """
        Embed a batch of documents and return both CLS and mean pooled embeddings.

        Args:
            batch: Dictionary containing 'input_ids', 'attention_mask', etc. for a batch of documents

        Returns:
            Dictionary with document IDs and their corresponding embeddings (both cls and mean)
        """
        # Get model outputs
        outputs = self.get_model_outputs(batch["input_ids"], batch["attention_mask"])
        last_hidden_state = outputs.last_hidden_state

        # Get both types of embeddings
        all_embeddings = self.get_embeddings_from_hidden_states(
            last_hidden_state, batch["attention_mask"].to(self.device)
        )

        # Create result dictionary with primary embeddings based on pooling strategy
        # and also include both types of embeddings
        result = {
            "ids": batch["id"],
            "cls_embeddings": all_embeddings["cls"].cpu(),
            "mean_embeddings": all_embeddings["mean"].cpu(),
        }

        return result

    def embed_dataloader(
        self, dataloader: DataLoader
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Embed all documents in a dataloader.

        Args:
            dataloader: PyTorch DataLoader containing batches of documents to embed

        Returns:
            Dictionary with "cls" and "mean" as top-level keys, each mapping document IDs to their
            corresponding embeddings
        """
        all_embeddings = {"cls": {}, "mean": {}}
        all_ids = []
        all_cls_embedding_tensors = []
        all_mean_embedding_tensors = []

        # Process each batch in the dataloader
        for batch in tqdm.tqdm(dataloader, desc="Embedding documents"):
            batch_result = self.embed_batch(batch)

            # Extract IDs and embeddings from batch result
            batch_ids = batch_result["ids"]
            batch_cls_embeddings = batch_result["cls_embeddings"]
            batch_mean_embeddings = batch_result["mean_embeddings"]

            # Add to lists
            all_ids.extend(batch_ids)
            all_cls_embedding_tensors.append(batch_cls_embeddings)
            all_mean_embedding_tensors.append(batch_mean_embeddings)

        # Concatenate all embedding tensors
        if all_cls_embedding_tensors:
            all_cls_embeddings_tensor = torch.cat(all_cls_embedding_tensors, dim=0)
            all_mean_embeddings_tensor = torch.cat(all_mean_embedding_tensors, dim=0)

            # Create mapping from IDs to embeddings
            for i, doc_id in enumerate(all_ids):
                all_embeddings["cls"][doc_id] = all_cls_embeddings_tensor[i]
                all_embeddings["mean"][doc_id] = all_mean_embeddings_tensor[i]

        return all_embeddings


class LateChunkingEmbedder(BaseEmbedder):
    """
    Class for creating embeddings using the late chunking approach.

    This class processes concatenated documents where multiple text segments
    are combined into a single input. It embeds the entire input at once, then
    extracts embeddings for each segment based on token boundaries.

    For each batch item, there is:
    - One CLS embedding (from the first token of the sequence)
    - One MEAN embedding (mean pooled from the entire sequence)
    - Multiple segment embeddings (one per text segment using mean pooling)
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        normalize: bool = True,
        jina_v3_task: Optional[str] = "text-matching",
        apply_attn_calibration: bool = False,
        calib_layers: Optional[str] = None,
        calib_source_tokens: Optional[str] = None,
        calib_basket_size: Optional[int] = None,
        calib_strength: Optional[float] = 1.0,
        device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
    ):
        """
        Initialize the LateChunkingEmbedder with a HuggingFace model.

        Args:
            model_name: Name or path of the HuggingFace model to use for embedding
            device: Device to run the model on ('cpu', 'cuda', 'mps').
                   If None, will use CUDA or MPS if available, else CPU.
            normalize: Whether to normalize the embeddings to unit length (L2 norm)
            jina_v3_task: Task type for Jina v3 embeddings model
        """
        super().__init__(
            model_name=model_name,
            device=device,
            normalize=normalize,
            jina_v3_task=jina_v3_task,
            apply_attn_calibration=apply_attn_calibration,
            calib_layers=calib_layers,
            calib_source_tokens=calib_source_tokens,
            calib_basket_size=calib_basket_size,
            calib_strength=calib_strength,
            device_map=device_map,
        )

    def embed_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[int, Dict[str, Dict[str, torch.Tensor]]]:
        """
        Embed a batch of concatenated documents and extract segment embeddings.

        Args:
            batch: Dictionary containing 'input_ids', 'attention_mask', 'doc_boundaries', etc.

        Returns:
            Dictionary mapping batch indices to sub-dictionaries containing:
            - One CLS embedding for the entire sequence
            - One MEAN embedding for the entire sequence
            - Multiple segment embeddings (one per document) using mean pooling
        """
        # Get model outputs
        outputs = self.get_model_outputs(batch["input_ids"], batch["attention_mask"])
        last_hidden_state = outputs.last_hidden_state

        # Process each item in the batch
        batch_results = {}
        for i in range(batch["input_ids"].size(0)):
            doc_boundaries = batch["doc_boundaries"][i]
            doc_ids = batch["doc_ids"][i]

            # Create a context identifier for this sequence of documents
            context_id = batch["id"][i]

            # Get the CLS token embedding (first token of the entire sequence)
            cls_embedding = last_hidden_state[i, 0]
            if self.normalize:
                cls_embedding = F.normalize(cls_embedding, p=2, dim=0)

            # Get the MEAN pooled embedding for the entire sequence
            attention_mask_for_item = (
                batch["attention_mask"][i].unsqueeze(0).to(self.device)
            )
            mean_embedding = self.mean_pooling(
                last_hidden_state[i].unsqueeze(0), attention_mask_for_item
            ).squeeze(0)
            if self.normalize:
                mean_embedding = F.normalize(mean_embedding, p=2, dim=0)

            # Create embeddings for each segment using document boundaries
            segment_embeddings = {}

            for j, (doc_id, boundary) in enumerate(zip(doc_ids, doc_boundaries)):
                # Extract start and end indices (start inclusive, end exclusive)
                start_idx, end_idx = boundary[0], boundary[1]

                # Extract segment tokens from the hidden state
                segment_hidden = last_hidden_state[i, start_idx:end_idx]

                # Skip empty segments
                if segment_hidden.shape[0] == 0:
                    continue

                # Apply mean pooling for the segment
                segment_embedding = torch.mean(segment_hidden, dim=0)

                # Normalize embedding if required
                if self.normalize:
                    segment_embedding = F.normalize(segment_embedding, p=2, dim=0)

                # Create a unique key for this document that includes the context
                unique_doc_key = f"{doc_id}__pos{j}__{context_id}"

                # Store embedding with the unique document ID
                segment_embeddings[unique_doc_key] = segment_embedding.cpu()

            batch_results[context_id] = {
                "cls": cls_embedding.cpu(),
                "mean": mean_embedding.cpu(),
                "segment_embeddings": segment_embeddings,
            }

        return batch_results

    def embed_dataloader(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Embed all documents in a dataloader using the late chunking approach.

        Args:
            dataloader: PyTorch DataLoader containing batches of concatenated documents

        Returns:
            Dictionary mapping:
            - Document IDs to their mean pooled segment embeddings
            - Special CLS keys to the CLS embeddings for each sequence
            - Special MEAN keys to the MEAN embeddings for each sequence
        """
        all_embeddings = {"cls": {}, "mean": {}, "segment_embeddings": {}}

        # Process each batch in the dataloader
        for batch in tqdm.tqdm(dataloader, desc="Embedding documents (late chunking)"):
            batch_results = self.embed_batch(batch)

            # Update the overall results dictionary with embeddings from this batch
            for context_id, batch_embeddings in batch_results.items():
                # Store the CLS embedding for this context
                all_embeddings["cls"][context_id] = batch_embeddings["cls"]
                # Store the MEAN embedding for this context
                all_embeddings["mean"][context_id] = batch_embeddings["mean"]
                # Store the segment embeddings for this context
                all_embeddings["segment_embeddings"].update(
                    batch_embeddings["segment_embeddings"]
                )

        return all_embeddings


class TextEmbedderOLD:
    """
    Lightweight SentenceTransformers-like wrapper focused on encode().

    Constraints/assumptions:
        - Only supports model_name == "Alibaba-NLP/gte-multilingual-base".
        - Uses model-appropriate pooling selected via an instance-level mapping
            (for Alibaba: CLS pooling). Normalization is controlled at class level.
    - Optional attention calibration only if the instance was created with
      apply_attn_calibration=True and required calibration settings.
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = True,
        *,
        device: Optional[str] = None,
        normalize: bool = True,
        apply_attn_calibration: bool = False,
        calib_layers: Optional[str] = None,
        calib_source_tokens: Optional[str] = None,
        calib_basket_size: Optional[int] = None,
        calib_strength: Optional[float] = 1.0,
        device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
    ) -> None:
        assert (
            model_name == "Alibaba-NLP/gte-multilingual-base"
        ), "TextEmbedder currently supports only 'Alibaba-NLP/gte-multilingual-base'"

        # Backend for model execution and pooling
        self._backend = BaseEmbedder(
            model_name=model_name,
            device=device,
            normalize=normalize,
            jina_v3_task="text-matching",
            apply_attn_calibration=apply_attn_calibration,
            calib_layers=calib_layers,
            calib_source_tokens=calib_source_tokens,
            calib_basket_size=calib_basket_size,
            calib_strength=calib_strength,
            device_map=device_map,
        )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )

        # Mirror a subset of SentenceTransformers attributes for familiarity
        self.model_name = model_name
        self.normalize = normalize

        # Instance-level mapping from model name to pooling method
        # Supported methods: "cls", "mean"
        self.pooling_by_model: Dict[str, str] = {
            "Alibaba-NLP/gte-multilingual-base": "cls",
        }

    def encode(
        self,
        sentences: Union[str, List[str]],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        normalize_embeddings: Optional[bool] = None,
        device: Optional[str] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Encode a sentence or list of sentences into embeddings.

        Inputs (subset compatible with SentenceTransformers.encode used in notebooks):
        - sentences: str | List[str]
        - batch_size: int (default 32)
        - show_progress_bar: bool
        - convert_to_numpy: bool (default True)
        - convert_to_tensor: bool (default False)
        - normalize_embeddings: ignored; normalization is controlled at class level

        Returns: embeddings as numpy array (N, D) by default or torch.Tensor when requested.
        """
        # Input normalization
        if isinstance(sentences, str):
            sentences_list: List[str] = [sentences]
        else:
            sentences_list = list(sentences)
        assert len(sentences_list) > 0, "No sentences to encode"

        # Device selection respects backend unless explicitly overridden
        use_device = device if device is not None else self._backend.device

        # Batch loop
        all_embeddings: List[torch.Tensor] = []
        rng = range(0, len(sentences_list), batch_size)
        iterator = tqdm.tqdm(rng, desc="Encoding", disable=not show_progress_bar)
        for start in iterator:
            end = min(start + batch_size, len(sentences_list))
            chunk = sentences_list[start:end]

            enc = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids: torch.Tensor = enc["input_ids"]
            attention_mask: torch.Tensor = enc["attention_mask"]
            assert input_ids.dim() == 2 and attention_mask.dim() == 2
            B, L = input_ids.shape
            assert attention_mask.shape == (B, L)
            # print(f"Processing batch of size {B} with sequence length {L}")

            # Move if not using device_map; BaseEmbedder.get_model_outputs handles this as well
            if self._backend.device_map is None:
                input_ids = input_ids.to(use_device)
                attention_mask = attention_mask.to(use_device)

            # Forward (calibration behavior is controlled at instance level in BaseEmbedder)
            outputs = self._backend.get_model_outputs(
                input_ids=input_ids, attention_mask=attention_mask
            )

            last_hidden_state: torch.Tensor = outputs.last_hidden_state
            assert last_hidden_state.dim() == 3 and last_hidden_state.shape[0] == B

            # Model-appropriate pooling
            pooling_method = self.pooling_by_model.get(self.model_name, "cls")
            if pooling_method == "cls":
                pooled = last_hidden_state[:, 0, :]
            elif pooling_method == "mean":
                pooled = self._backend.mean_pooling(last_hidden_state, attention_mask)
            else:
                raise AssertionError(f"Unsupported pooling method: {pooling_method}")
            assert (
                pooled.dim() == 2 and pooled.shape[0] == B
            ), f"Pooled shape mismatch: {pooled.shape}, expected (B, D) with B={B}"

            # Normalize at class level
            if self.normalize:
                pooled = F.normalize(pooled, p=2, dim=1)

            # Accumulate on CPU to match typical SentenceTransformers behavior
            all_embeddings.append(pooled.cpu())

        # Concat
        embeddings = torch.cat(all_embeddings, dim=0)

        # Output formatting
        if convert_to_tensor:
            return embeddings
        if convert_to_numpy:
            return embeddings.detach().numpy()
        return embeddings


class FairSentenceTransformer(SentenceTransformer):
    """SentenceTransformers-compatible embedder with positional fairness calibration."""

    TESTED_MODELS: Tuple[str, ...] = ("Alibaba-NLP/gte-multilingual-base",)

    DECODER_ARCH_NAMES: Set[str] = {"qwen", "llama", "gpt", "mistral", "falcon"}

    ATTENTION_PATHS: Dict[str, Union[str, Callable[[Any, int], Any]]] = {
        "Alibaba-NLP/gte-multilingual-base": "encoder.layer[{i}].attention.source.self__attention_0.source.nn_functional_softmax_0.output",
        "BAAI/bge-m3": "encoder.layer[{i}].attention.self.source.nn_functional_softmax_0.output",
        "Qwen/Qwen3-Embedding-0.6B": "layers[{i}].self_attn.source.attention_interface_0.source.nn_functional_softmax_0.output",
        "Qwen/Qwen3-Embedding-4B": "layers[{i}].self_attn.source.attention_interface_0.source.nn_functional_softmax_0.output",
        "Qwen/Qwen3-Embedding-8B": "layers[{i}].self_attn.source.attention_interface_0.source.nn_functional_softmax_0.output",
        "jinaai/jina-embeddings-v3": "roberta.encoder.layers[{i}].mixer.inner_attn.source.torch_softmax_0.output",
        "NovaSearch/stella_en_400M_v5": "encoder.layer[{i}].attention.source.self__attention_0.source.nn_functional_softmax_0.output",
    }

    ############################################################################
    #                               NOTES                                      #
    ############################################################################
    # - For NovaSearch/stella_en_400M_v5, the following changes need to be made in modeling.py:
    #   1. change line 684: from attn_implementation=None to attn_implementation="eager"
    #   2. change line 689: from != to ==
    #   3. add new line 940: unpad_inputs = False

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: Optional[str] = None,
        device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
        attention_path_override: Optional[Callable[[Any, int], Any]] = None,
        pooling_override: Optional[Literal["cls", "mean", "last"]] = None,
        calibrated_tokens_override: Optional[Literal["cls", "all", "last"]] = None,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> None:
        self._torch_dtype = (
            torch.float16
            if torch.cuda.is_available() or torch.backends.mps.is_available()
            else None
        )
        self._config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        self.padding_side = self._detect_padding_side()

        original_class_name = self.__class__.__name__
        self.__class__.__name__ = "SentenceTransformer"
        try:
            super().__init__(
                model_name_or_path,
                device=device,
                trust_remote_code=trust_remote_code,
                model_kwargs={
                    "device_map": device_map,
                    "torch_dtype": self._torch_dtype,
                },
                tokenizer_kwargs={"padding_side": self.padding_side},
                **kwargs,
            )
        finally:
            self.__class__.__name__ = original_class_name

        self.model_name_or_path = model_name_or_path
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self._attention_path_override = attention_path_override
        self._pooling_override = pooling_override
        self._calibrated_tokens_override = calibrated_tokens_override
        if model_name_or_path not in self.TESTED_MODELS:
            warnings.warn(
                f"Model '{model_name_or_path}' is untested; tested models: {self.TESTED_MODELS}",
                RuntimeWarning,
            )

        self.pooling_strategy = self._infer_pooling_strategy()
        self.calib_source_mode = self._calibration_source_mode()

        self._nnsight_model: Optional[nnsight.NNsight] = None
        self._num_layers: Optional[int] = None
        print(
            "Summary of TextEmbedder configuration:"
            f"\n  Model: {model_name_or_path}"
            f"\n  Padding side: {self.padding_side}"
            f"\n  Pooling strategy: {self.pooling_strategy}"
            f"\n  Calibrated token(s): {self.calib_source_mode}"
        )

    def _detect_padding_side(self) -> Literal["left", "right"]:
        model_type = str(getattr(self._config, "model_type", "")).lower()
        if any(name in model_type for name in self.DECODER_ARCH_NAMES):
            return "left"
        return "right"

    def _infer_pooling_strategy(self) -> Literal["cls", "mean", "last"]:
        """Infer pooling strategy from ST pooling module; fallback to CLS."""
        override = self._pooling_override
        if override is not None:
            if override in ("cls", "mean", "last"):
                return override
            raise AssertionError(f"Unsupported pooling override: {override}")

        pooling_modules = [m for m in self._modules.values() if isinstance(m, Pooling)]
        if pooling_modules:
            pooling = pooling_modules[0]
            if getattr(pooling, "pooling_mode_lasttoken", False):
                return "last"
            if getattr(pooling, "pooling_mode_cls_token", False):
                return "cls"
            if getattr(pooling, "pooling_mode_mean_tokens", False):
                return "mean"
        warnings.warn(
            "Pooling strategy could not be inferred; defaulting to first token (CLS)",
            RuntimeWarning,
        )
        return "cls"

    def _calibration_source_mode(self) -> Literal["cls", "all", "last"]:
        """Determine which token(s) to use for attention calibration."""
        override = self._calibrated_tokens_override
        if override is not None:
            if override in ("cls", "all", "last"):
                return override
            raise AssertionError(f"Unsupported calibrated tokens override: {override}")

        if self.pooling_strategy == "cls":
            return "cls"
        if self.pooling_strategy == "mean":
            return "all"
        if self.pooling_strategy == "last":
            return "last"
        raise AssertionError(f"Unsupported pooling strategy: {self.pooling_strategy}")

    def _resolve_attention_softmax(self, layer_idx: int):
        assert self._nnsight_model is not None
        resolver = self._attention_path_override
        if resolver is None:
            resolver = self._default_attention_path(self.model_name_or_path)
        assert (
            resolver is not None
        ), f"No attention path resolver for {self.model_name_or_path}"
        return resolver(self._nnsight_model, layer_idx)

    @staticmethod
    def _compile_attention_path(path_template: str) -> Callable[[Any, int], Any]:
        def resolver(model: Any, layer_idx: int) -> Any:
            path = path_template.format(i=layer_idx)
            target = model
            for segment in path.split("."):
                while "[" in segment:
                    attr, bracket, rest = segment.partition("[")
                    assert bracket == "["
                    index_str, closing, remainder = rest.partition("]")
                    assert closing == "]"
                    if attr:
                        target = getattr(target, attr)
                    idx = int(index_str)
                    target = target[idx]
                    segment = remainder
                if segment:
                    target = getattr(target, segment)
            return target

        return resolver

    def _default_attention_path(
        self, model_name: str
    ) -> Optional[Callable[[Any, int], Any]]:
        entry = self.ATTENTION_PATHS.get(model_name)
        if entry is None:
            return None
        if isinstance(entry, str):
            return self._compile_attention_path(entry)
        return entry

    def _ensure_nnsight_model(self) -> None:
        if self._nnsight_model is not None:
            return
        hf_model = AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            attn_implementation="eager",
            device_map=self.device_map,
            torch_dtype=self._torch_dtype,
        )
        if self.device_map is None:
            hf_model = hf_model.to(self.device)
        self._num_layers = int(hf_model.config.num_hidden_layers)
        self._nnsight_model = nnsight.NNsight(hf_model)

    def _validate_padding_mask(
        self, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.padding_side in ("left", "right")
        valid_len = attention_mask.sum(dim=1)
        K = attention_mask.size(1)
        pos = torch.arange(K, device=attention_mask.device).unsqueeze(0)
        if self.padding_side == "right":
            expected = (pos < valid_len.unsqueeze(1)).to(attention_mask.dtype)
            start_idx = torch.zeros_like(valid_len, dtype=torch.long)
        else:
            start_idx = (K - valid_len).clamp_min(0).to(torch.long)
            expected = (pos >= start_idx.unsqueeze(1)).to(attention_mask.dtype)
        assert torch.equal(
            attention_mask, expected
        ), f"Attention mask must be {self.padding_side}-padded"
        return expected.to(dtype=torch.bool), start_idx, valid_len.to(torch.long)

    def _calibrate_attention(
        self,
        attn: torch.Tensor,
        attention_mask: torch.Tensor,
        basket_size: int,
        strength: float,
        source_mode: Literal["cls", "all", "last"],
    ) -> torch.Tensor:
        attention_mask = attention_mask.to(attn.device)
        assert attn.dim() == 4, f"Expected attention 4D, got {attn.shape}"
        B, H, Q, K = attn.shape
        assert attention_mask.shape == (B, K)
        assert Q == K, "Self-attention expected (Q==K)"
        assert basket_size > 0
        valid_mask, start_idx, valid_len = self._validate_padding_mask(attention_mask)
        assert torch.all(valid_len > 0)
        valid_mask = valid_mask.to(device=attn.device)
        start_idx = start_idx.to(device=attn.device)
        valid_len = valid_len.to(device=attn.device)

        if source_mode == "cls":
            pool_pos = start_idx
            q_mask = torch.zeros((B, Q), device=attn.device, dtype=torch.bool)
            q_mask[torch.arange(B, device=attn.device), pool_pos] = True
        elif source_mode == "all":
            pool_pos = start_idx
            q_mask = valid_mask
        elif source_mode == "last":
            pool_pos = start_idx + valid_len - 1
            q_mask = torch.zeros((B, Q), device=attn.device, dtype=torch.bool)
            q_mask[torch.arange(B, device=attn.device), pool_pos] = True
        else:
            raise AssertionError(f"Unsupported source_mode: {source_mode}")
        q_mask_b = q_mask.view(B, 1, Q, 1)

        S = int(basket_size)
        pos = torch.arange(K, device=attn.device).view(1, 1, 1, K)
        start_idx_b = start_idx.view(B, 1, 1, 1)
        content_rel = (pos - start_idx_b).clamp_min(0)

        if source_mode == "all":
            basket_ids = content_rel // S
            max_baskets = int(((valid_len + S - 1) // S).max().item())
            basket_ids = basket_ids.clamp_max(max_baskets - 1)
            n_baskets = ((valid_len + S - 1) // S).to(dtype=attn.dtype).view(B, 1, 1, 1)
        else:
            pool_pos_b = pool_pos.view(B, 1, 1, 1)
            basket_ids = torch.where(
                pos == pool_pos_b,
                torch.zeros_like(pos),
                1 + ((content_rel - 1).clamp_min(0) // S),
            )
            max_content_baskets = ((valid_len - 1 + (S - 1)) // S).clamp_min(0)
            max_baskets = 1 + int(max_content_baskets.max().item())
            basket_ids = basket_ids.clamp_max(max_baskets - 1)
            n_content = ((torch.clamp_min(valid_len - 1, 0) + (S - 1)) // S).to(
                dtype=attn.dtype
            )
            n_baskets = (1 + n_content).view(B, 1, 1, 1)

        basket_ids = basket_ids.expand(B, H, Q, K)

        mask_bk = valid_mask.view(B, 1, 1, K).to(dtype=attn.dtype)
        attn_masked = attn * mask_bk

        bucket_sums = torch.zeros(
            (B, H, Q, max_baskets), device=attn.device, dtype=attn.dtype
        )
        bucket_sums = bucket_sums.scatter_add(
            dim=-1, index=basket_ids.to(torch.long), src=attn_masked
        )
        denom = bucket_sums.gather(dim=-1, index=basket_ids.to(torch.long))

        scaled = attn_masked * S
        calibrated = torch.where(
            denom > 0, (scaled / denom) * (1.0 / n_baskets), torch.zeros_like(scaled)
        )
        calibrated = calibrated * mask_bk

        row_sum = calibrated.sum(dim=-1, keepdim=True)
        row_sum_squeezed = row_sum.squeeze(-1)
        must_be_pos = q_mask.view(B, 1, Q).expand(B, H, Q)
        assert torch.all(
            row_sum_squeezed[must_be_pos] > 0
        ), "Calibrated rows must sum to >0"

        denom_rows = torch.where(q_mask_b, row_sum, torch.ones_like(row_sum))
        normalized = torch.where(q_mask_b, calibrated / denom_rows, attn_masked)

        orig_masked = attn * mask_bk
        blended = torch.where(
            q_mask_b,
            normalized * strength + orig_masked * (1.0 - strength),
            orig_masked,
        )
        return blended

    def _extract_last_hidden_state(self, model_output: Any) -> torch.Tensor:
        if hasattr(model_output, "last_hidden_state"):
            return model_output.last_hidden_state
        if isinstance(model_output, dict):
            assert "last_hidden_state" in model_output, "Missing last_hidden_state"
            return model_output["last_hidden_state"]
        assert isinstance(model_output, (list, tuple)) and len(model_output) > 0
        return model_output[0]

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attention_mask = attention_mask.to(hidden.device)
        valid_mask, start_idx, valid_len = self._validate_padding_mask(attention_mask)
        if self.pooling_strategy == "cls":
            pooled = hidden[
                torch.arange(hidden.size(0), device=hidden.device), start_idx
            ]
        elif self.pooling_strategy == "mean":
            pooled = self._mean_pool(hidden, attention_mask)
        elif self.pooling_strategy == "last":
            last_idx = start_idx + valid_len - 1
            pooled = hidden[
                torch.arange(hidden.size(0), device=hidden.device), last_idx
            ]
        else:
            raise AssertionError(
                f"Unsupported pooling strategy: {self.pooling_strategy}"
            )
        assert pooled.dim() == 2 and pooled.size(0) == hidden.size(0)
        return pooled

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        lengths = mask.sum(dim=1)
        return summed / lengths

    def encode_positionally_fair(
        self,
        sentences: Union[str, List[str]],
        *,
        calib_basket_size: int,
        calib_layers: int,
        calib_strength: float,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        device: Optional[str] = None,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        assert calib_basket_size > 0
        assert calib_layers > 0
        assert 0.0 <= calib_strength <= 1.0

        self._ensure_nnsight_model()
        assert self._nnsight_model is not None and self._num_layers is not None
        assert calib_layers <= self._num_layers, "calib_layers exceeds model depth"

        if isinstance(sentences, str):
            sentences_list: List[str] = [sentences]
        else:
            sentences_list = list(sentences)
        assert len(sentences_list) > 0

        use_device = (
            torch.device(device) if device is not None else torch.device(self.device)
        )

        all_embeddings: List[torch.Tensor] = []
        rng = range(0, len(sentences_list), batch_size)
        iterator = tqdm.tqdm(rng, desc="Encoding (fair)", disable=not show_progress_bar)
        for start in iterator:
            end = min(start + batch_size, len(sentences_list))
            chunk = sentences_list[start:end]
            enc = self.tokenizer(
                chunk, padding=True, truncation=True, return_tensors="pt"
            )
            input_ids: torch.Tensor = enc["input_ids"]
            attention_mask: torch.Tensor = enc["attention_mask"]
            assert input_ids.dim() == 2 and attention_mask.dim() == 2
            B, L = input_ids.shape
            assert attention_mask.shape == (B, L)

            if self.device_map is None:
                input_ids = input_ids.to(use_device)
                attention_mask = attention_mask.to(use_device)

            with self._nnsight_model.trace() as tracer:
                with tracer.invoke(input_ids=input_ids, attention_mask=attention_mask):
                    layer_start = max(0, self._num_layers - calib_layers)
                    for idx in range(layer_start, self._num_layers):
                        attn = self._resolve_attention_softmax(idx)
                        calibrated = self._calibrate_attention(
                            attn,
                            attention_mask,
                            basket_size=calib_basket_size,
                            strength=calib_strength,
                            source_mode=self.calib_source_mode,
                        )
                        attn.copy_(calibrated)
                    model_output = self._nnsight_model.output.save()

            hidden = self._extract_last_hidden_state(model_output)
            pooled = self._pool(hidden, attention_mask)
            if normalize_embeddings:
                pooled = F.normalize(pooled, p=2, dim=1)
            all_embeddings.append(pooled.cpu())

        embeddings = torch.cat(all_embeddings, dim=0)
        if convert_to_tensor:
            return embeddings
        if convert_to_numpy:
            return embeddings.detach().numpy()
        return embeddings
