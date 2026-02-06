"""
Attention analysis for long concatenated documents.

This module reuses the existing dataset + indices pipeline to build a
concatenated dataloader, runs the encoder model with output_attentions=True,
and aggregates incoming attention per destination position:

- Basket-level: aggregate per fixed-size token baskets (e.g., 64/128/256)
- Article-level: aggregate per article using doc_boundaries
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoModel
from tqdm import tqdm

from fair_sentence_transformers.core.document_handler import DocumentHandler


@dataclass
class AnalysisConfig:
    config_path: str
    analysis_mode: str  # "baskets" | "articles"
    basket_size: int = 128
    rel_bins: int = 20
    exclude_first_token: bool = True
    exclude_last_token: bool = True
    max_examples: Optional[int] = None
    batch_size: Optional[int] = None
    device: Optional[str] = None
    only_from_first_token: bool = False
    compute_maps: bool = True
    compute_incoming: bool = True


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def build_concat_loader_from_config(
    config: Dict[str, Any],
) -> Tuple[torch.utils.data.DataLoader, Dict[str, Any]]:
    model_name = config["model_name"]
    tokenized_dataset_path = config["tokenized_dataset_path"]
    indices_path = config["indices_path"]
    separator = config.get("separator", " ")
    source_lang = config.get("source_lang", "en")
    target_lang = config.get("target_lang")
    batch_size_concat = config.get("batch_size_concat", 1)

    tokenized = load_from_disk(tokenized_dataset_path)
    assert isinstance(tokenized, (Dataset, DatasetDict))

    indices = _load_json(indices_path)
    assert "concat_indices" in indices
    concat_indices: List[List[int]] = indices["concat_indices"]
    assert isinstance(concat_indices, list) and len(concat_indices) > 0

    doc_handler = DocumentHandler(tokenizer_name=model_name)

    if isinstance(tokenized, DatasetDict):
        # wiki_parallel path
        datasets, _, _ = doc_handler.prepare_datasets_wiki_parallel(
            dataset_dict=tokenized,
            concat_indices=concat_indices,
            separator=separator,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        concat_dataset = datasets["concatenated"]
    else:
        # Generic concatenation on a monolingual Dataset
        concat_dataset = doc_handler.create_concatenated_dataset(
            dataset=tokenized,
            concat_indices=concat_indices,
            separator=separator,
        )

    concat_loader = doc_handler.get_dataloader(
        concat_dataset, batch_size=batch_size_concat, shuffle=False
    )
    return concat_loader, {
        "model_name": model_name,
        "separator": separator,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "indices_path": indices_path,
        "dataset_path": tokenized_dataset_path,
    }


class AttentionAggregator:
    def __init__(
        self,
        model_name: str,
        device: Optional[str],
        analysis_mode: str,
        basket_size: int,
        rel_bins: int,
        exclude_first_token: bool,
        exclude_last_token: bool,
        only_from_first_token: bool,
        compute_maps: bool,
        compute_incoming: bool,
    ) -> None:
        assert analysis_mode in ("baskets", "articles")

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        self.device = device

        if model_name == "jinaai/jina-embeddings-v3":
            model_name = "jinaai/jina-embeddings-v2-base-en"

        # Prefer lower precision on accelerator to cut memory
        torch_dtype = None
        if isinstance(self.device, str) and self.device.startswith("cuda"):
            torch_dtype = torch.bfloat16
        elif isinstance(self.device, str) and self.device.startswith("mps"):
            torch_dtype = torch.float16

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype=torch_dtype,
        )
        self.model.eval()
        self.model.to(self.device)
        self.model_name = model_name

        self.analysis_mode = analysis_mode
        self.basket_size = basket_size
        self.rel_bins = rel_bins
        self.exclude_first_token = exclude_first_token
        self.exclude_last_token = exclude_last_token
        self.only_from_first_token = only_from_first_token
        self.compute_maps = compute_maps
        self.compute_incoming = compute_incoming

        # Aggregation state
        self.num_layers: Optional[int] = None
        self.num_abs_bins: Optional[int] = None
        self.num_rel_bins: Optional[int] = None
        self.num_articles: Optional[int] = None
        self.seq_len_ref: Optional[int] = None
        self.max_total_bins: Optional[int] = None

        # Absolute baskets accumulators [layers, bins]
        self.sum_abs: Optional[torch.Tensor] = None
        self.sum_abs_mass: Optional[torch.Tensor] = None
        self.count_abs: Optional[torch.Tensor] = None
        # Relative baskets accumulators [layers, bins]
        self.sum_rel: Optional[torch.Tensor] = None
        self.sum_rel_mass: Optional[torch.Tensor] = None
        self.count_rel: Optional[torch.Tensor] = None

        # Special token accumulators [layers]
        self.sum_first: Optional[torch.Tensor] = None
        self.count_first: Optional[torch.Tensor] = None
        self.sum_last: Optional[torch.Tensor] = None
        self.count_last: Optional[torch.Tensor] = None

        # Layer-wise bin attention matrices [layers, bins, bins]
        self.layer_bin_mass_sum: Optional[torch.Tensor] = None
        self.layer_bin_count: Optional[torch.Tensor] = None

        # Basket-aggregated attention map accumulators (averaged over heads and layers)
        # Absolute (dynamic number of bins): [B_max, B_max]
        self.map_abs_sum = None
        self.map_abs_count = None
        # Relative (fixed rel_bins): [rel_bins, rel_bins]
        self.map_rel_sum = None
        self.map_rel_count = None

        self.examples_seen = 0

    @torch.no_grad()
    def process_batch(self, batch: Dict[str, Any]) -> None:
        input_ids: torch.Tensor = batch["input_ids"].to(self.device)
        attention_mask: torch.Tensor = batch["attention_mask"].to(self.device)

        assert input_ids.dim() == 2 and attention_mask.shape == input_ids.shape
        bsz, seq_len = input_ids.shape

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )
        attentions = outputs.attentions
        # Drop the container to avoid holding extra references on GPU
        del outputs
        assert isinstance(attentions, (list, tuple)) and len(attentions) > 0

        num_layers = len(attentions)
        num_heads = attentions[0].shape[1]
        for l in range(num_layers):
            a = attentions[l]
            assert a.shape == (bsz, num_heads, seq_len, seq_len)

        if self.num_layers is None:
            self.num_layers = num_layers
        else:
            assert self.num_layers == num_layers

        # For each example in batch, compute reduced incoming attention vector per layer (optional)
        for i in range(bsz):
            mask_i = attention_mask[i]  # [L]
            valid_idx = mask_i.nonzero(as_tuple=False).squeeze(-1)
            assert valid_idx.dim() == 1
            assert valid_idx.numel() >= 2

            # Build effective index list consistent with exclusions for map computation
            eff_idx = valid_idx
            if self.exclude_first_token:
                eff_idx = eff_idx[1:]
            if self.exclude_last_token:
                eff_idx = eff_idx[:-1]
            L_eff = eff_idx.numel()
            # If exclusions removed all targets, skip
            if L_eff == 0:
                self.examples_seen += 1
                continue

            # Build bin definitions including first and last tokens explicitly
            row_bins_indices: List[torch.Tensor] = []
            first_idx = valid_idx[0:1]
            last_idx = valid_idx[-1:].clone()
            row_bins_indices.append(first_idx)

            middle_idx = valid_idx[1:-1]
            middle_count = middle_idx.numel()
            if middle_count > 0:
                num_middle_bins = (
                    middle_count + self.basket_size - 1
                ) // self.basket_size
                for b in range(num_middle_bins):
                    s_mid = b * self.basket_size
                    e_mid = min((b + 1) * self.basket_size, middle_count)
                    seg_idx = middle_idx[s_mid:e_mid]
                    assert seg_idx.numel() > 0
                    row_bins_indices.append(seg_idx)

            row_bins_indices.append(last_idx)
            total_bins = len(row_bins_indices)
            if self.max_total_bins is None or total_bins > self.max_total_bins:
                self.max_total_bins = total_bins
            assert self.max_total_bins is not None
            self._ensure_layer_bin_buffers(self.max_total_bins)

            # Build per-layer vector per valid target position
            # Default: incoming[j] = sum over queries of A[query->key=j]
            # If only_from_first_token: use only query at first valid position
            per_layer_vecs: Optional[List[torch.Tensor]] = None
            if self.compute_incoming:
                per_layer_vecs = []  # each length valid_idx
                L_valid = valid_idx.numel()
                for l in range(num_layers):
                    a = attentions[l][i]  # [H, L, L]
                    if not self.only_from_first_token:
                        incoming = a.sum(dim=1)  # [H, L]
                        incoming_valid = incoming[:, valid_idx]  # [H, L_valid]
                        vec = incoming_valid.mean(dim=0)  # [L_valid]
                    else:
                        q_pos = int(valid_idx[0].item())
                        row = a[:, q_pos, :]  # [H, L]
                        row_valid = row[:, valid_idx]  # [H, L_valid]
                        vec = row_valid.mean(dim=0)  # [L_valid]
                    assert vec.shape == (L_valid,)

                    # Accumulate first/last token attention per layer (before trimming)
                    first_val = float(vec[0].item())
                    last_val = float(vec[-1].item())
                    if self.sum_first is None:
                        self.sum_first = torch.zeros(num_layers, dtype=torch.float32)
                        self.count_first = torch.zeros(num_layers, dtype=torch.float32)
                        self.sum_last = torch.zeros(num_layers, dtype=torch.float32)
                        self.count_last = torch.zeros(num_layers, dtype=torch.float32)
                    self.sum_first[l] += first_val
                    self.count_first[l] += 1.0
                    self.sum_last[l] += last_val
                    self.count_last[l] += 1.0

                    per_layer_vecs.append(vec)

                    del a

                # Apply exclusions to vectors
                if self.exclude_first_token:
                    per_layer_vecs = [v[1:] for v in per_layer_vecs]
                if self.exclude_last_token:
                    per_layer_vecs = [v[:-1] for v in per_layer_vecs]
                # Ensure trimmed length matches eff_idx
                assert per_layer_vecs[0].shape[0] == L_eff

            if self.analysis_mode == "baskets":
                if self.compute_incoming:
                    assert per_layer_vecs is not None
                    self._accumulate_baskets_absolute(per_layer_vecs)
                    # self._accumulate_baskets_relative(per_layer_vecs)
                # Optionally compute attention maps
                if self.compute_maps:
                    # Precompute bin ranges once per example
                    num_bins_abs = (L_eff + self.basket_size - 1) // self.basket_size
                    abs_bins: List[Tuple[int, int]] = []
                    for b in range(num_bins_abs):
                        rs = b * self.basket_size
                        re = min((b + 1) * self.basket_size, L_eff)
                        if re > rs:
                            abs_bins.append((rs, re))

                    # Relative bins
                    assert L_eff >= self.rel_bins
                    edges = torch.linspace(0, L_eff, steps=self.rel_bins + 1)
                    edges = torch.floor(edges).to(torch.int64)
                    edges[-1] = L_eff
                    for k in range(1, edges.numel()):
                        if edges[k] <= edges[k - 1]:
                            edges[k] = min(edges[k - 1] + 1, L_eff)
                    rel_bins: List[Tuple[int, int]] = []
                    for b in range(self.rel_bins):
                        rs = int(edges[b].item())
                        re = int(edges[b + 1].item())
                        if re > rs:
                            rel_bins.append((rs, re))

                    # Stream per-bin map means directly from attentions; one sync per bin pair
                    self._accumulate_maps_absolute_stream(
                        attentions, i, eff_idx, num_layers, num_heads, abs_bins
                    )
                    self._accumulate_maps_relative_stream(
                        attentions, i, eff_idx, num_layers, num_heads, rel_bins
                    )
            else:
                # Article-level using doc_boundaries
                if not self.compute_incoming:
                    # Articles require incoming vectors; maps are not defined here
                    self.examples_seen += 1
                    continue
                raw_bounds = batch["doc_boundaries"][i]
                # Convert to tensor deterministically; fail fast on unknown types
                if isinstance(raw_bounds, torch.Tensor):
                    doc_bounds = raw_bounds.to(valid_idx.device)
                elif isinstance(raw_bounds, (list, tuple)):
                    doc_bounds = torch.tensor(
                        raw_bounds, dtype=torch.long, device=valid_idx.device
                    )
                else:
                    try:
                        import numpy as _np  # local import to avoid global dependency if unused

                        if isinstance(raw_bounds, _np.ndarray):
                            doc_bounds = torch.from_numpy(raw_bounds).to(
                                valid_idx.device
                            )
                        else:
                            raise AssertionError(
                                f"Unsupported doc_boundaries type: {type(raw_bounds)}"
                            )
                    except Exception:
                        raise AssertionError(
                            f"Unsupported doc_boundaries type: {type(raw_bounds)}"
                        )

                assert (
                    doc_bounds.ndim == 2
                    and doc_bounds.shape[1] == 2
                    and doc_bounds.shape[0] > 0
                )

                # Convert doc_bounds from absolute positions to reduced index space
                # valid_idx maps reduced positions -> absolute indices.
                # We need to map absolute spans to reduced spans by locating their
                # start/end within valid_idx.
                spans: List[Tuple[int, int]] = []
                for s_abs, e_abs in doc_bounds.tolist():
                    # Find positions within valid_idx (assumes contiguous spans)
                    pos_start = (valid_idx == s_abs).nonzero(as_tuple=False)
                    pos_end = (valid_idx == (e_abs - 1)).nonzero(as_tuple=False)
                    assert pos_start.numel() == 1 and pos_end.numel() == 1
                    s_red = int(pos_start.item())
                    e_red = int(pos_end.item()) + 1
                    # If we excluded first token earlier from per_layer_vecs,
                    # shift spans into the trimmed index space and clamp
                    if self.exclude_first_token:
                        s_red = max(0, s_red - 1)
                        e_red = max(0, e_red - 1)
                    # If we excluded last token earlier, clamp end to new length
                    if self.exclude_last_token:
                        e_red = min(e_red, L_eff)
                    spans.append((s_red, e_red))

                if self.num_articles is None:
                    self.num_articles = len(spans)
                else:
                    assert self.num_articles == len(spans)

                assert per_layer_vecs is not None
                self._accumulate_articles(per_layer_vecs, spans)

            # Aggregate per-layer attention matrices over bin pairs
            assert self.layer_bin_mass_sum is not None
            assert self.layer_bin_count is not None
            for l in range(num_layers):
                layer_attn = attentions[l][i]
                assert layer_attn.shape == (num_heads, seq_len, seq_len)
                avg_attn = layer_attn.mean(dim=0)
                assert avg_attn.shape == (seq_len, seq_len)
                for row_idx, rows in enumerate(row_bins_indices):
                    assert rows.dim() == 1 and rows.numel() > 0
                    row_slice = torch.index_select(avg_attn, 0, rows)
                    row_token_count = int(rows.numel())
                    assert row_token_count > 0
                    for col_idx, cols in enumerate(row_bins_indices):
                        assert cols.dim() == 1 and cols.numel() > 0
                        block = torch.index_select(row_slice, 1, cols)
                        block_mass = float(block.sum().item()) / float(row_token_count)
                        self.layer_bin_mass_sum[l, row_idx, col_idx] += block_mass
                        self.layer_bin_count[l, row_idx, col_idx] += 1.0

            self.examples_seen += 1

    def _ensure_abs_buffers(self, L_bins: int) -> None:
        assert self.num_layers is not None
        if self.sum_abs is None:
            self.sum_abs = torch.zeros(self.num_layers, L_bins, dtype=torch.float32)
            self.sum_abs_mass = torch.zeros(
                self.num_layers, L_bins, dtype=torch.float32
            )
            self.count_abs = torch.zeros(self.num_layers, L_bins, dtype=torch.float32)
        elif self.sum_abs.shape[1] < L_bins:
            pad = L_bins - self.sum_abs.shape[1]
            assert self.sum_abs_mass is not None
            assert self.count_abs is not None
            self.sum_abs = torch.cat(
                [self.sum_abs, torch.zeros(self.num_layers, pad, dtype=torch.float32)],
                dim=1,
            )
            self.sum_abs_mass = torch.cat(
                [
                    self.sum_abs_mass,
                    torch.zeros(self.num_layers, pad, dtype=torch.float32),
                ],
                dim=1,
            )
            self.count_abs = torch.cat(
                [
                    self.count_abs,
                    torch.zeros(self.num_layers, pad, dtype=torch.float32),
                ],
                dim=1,
            )

    def _ensure_rel_buffers(self, L_bins: int) -> None:
        assert self.num_layers is not None
        if self.sum_rel is None:
            self.sum_rel = torch.zeros(self.num_layers, L_bins, dtype=torch.float32)
            self.sum_rel_mass = torch.zeros(
                self.num_layers, L_bins, dtype=torch.float32
            )
            self.count_rel = torch.zeros(self.num_layers, L_bins, dtype=torch.float32)
        else:
            assert self.sum_rel.shape == (self.num_layers, L_bins)
            assert self.count_rel.shape == (self.num_layers, L_bins)
            assert self.sum_rel_mass is not None
            assert self.sum_rel_mass.shape == (self.num_layers, L_bins)

    def _ensure_layer_bin_buffers(self, bins: int) -> None:
        assert self.num_layers is not None
        if self.layer_bin_mass_sum is None:
            self.layer_bin_mass_sum = torch.zeros(
                self.num_layers, bins, bins, dtype=torch.float32
            )
            self.layer_bin_count = torch.zeros(
                self.num_layers, bins, bins, dtype=torch.float32
            )
            return

        assert self.layer_bin_count is not None
        current_bins = self.layer_bin_mass_sum.shape[1]
        if current_bins < bins:
            new_sum = torch.zeros(self.num_layers, bins, bins, dtype=torch.float32)
            new_sum[:, :current_bins, :current_bins] = self.layer_bin_mass_sum
            self.layer_bin_mass_sum = new_sum

            new_count = torch.zeros(self.num_layers, bins, bins, dtype=torch.float32)
            new_count[:, :current_bins, :current_bins] = self.layer_bin_count
            self.layer_bin_count = new_count
        else:
            assert self.layer_bin_mass_sum.shape[0] == self.num_layers
            assert self.layer_bin_mass_sum.shape[1] >= bins
            assert self.layer_bin_mass_sum.shape[2] >= bins
            assert self.layer_bin_count.shape == self.layer_bin_mass_sum.shape

    def _ensure_map_abs_buffers(self, bins: int) -> None:
        if self.map_abs_sum is None:
            self.map_abs_sum = torch.zeros(bins, bins, dtype=torch.float32)
            self.map_abs_count = torch.zeros(bins, bins, dtype=torch.float32)
        else:
            B_old = self.map_abs_sum.shape[0]
            if bins > B_old:
                pad = bins - B_old
                pad_rows = torch.zeros(pad, B_old, dtype=torch.float32)
                pad_cols = torch.zeros(bins, pad, dtype=torch.float32)
                # expand sum
                self.map_abs_sum = torch.cat([self.map_abs_sum, pad_rows], dim=0)
                self.map_abs_sum = torch.cat([self.map_abs_sum, pad_cols], dim=1)
                # expand count
                self.map_abs_count = torch.cat(
                    [self.map_abs_count, pad_rows.clone()], dim=0
                )
                self.map_abs_count = torch.cat(
                    [self.map_abs_count, pad_cols.clone()], dim=1
                )

    def _ensure_map_rel_buffers(self) -> None:
        if self.map_rel_sum is None:
            self.map_rel_sum = torch.zeros(
                self.rel_bins, self.rel_bins, dtype=torch.float32
            )
            self.map_rel_count = torch.zeros(
                self.rel_bins, self.rel_bins, dtype=torch.float32
            )
        else:
            assert self.map_rel_sum.shape == (self.rel_bins, self.rel_bins)
            assert self.map_rel_count.shape == (self.rel_bins, self.rel_bins)

    def _accumulate_baskets_absolute(self, per_layer_vecs: List[torch.Tensor]) -> None:
        # per_layer_vecs: list of [L_eff]
        L_eff = per_layer_vecs[0].shape[0]
        for v in per_layer_vecs:
            assert v.shape == (L_eff,)

        # Determine number of absolute bins for this example
        num_bins = (L_eff + self.basket_size - 1) // self.basket_size
        if self.num_abs_bins is None or num_bins > self.num_abs_bins:
            self.num_abs_bins = num_bins
        self._ensure_abs_buffers(self.num_abs_bins)

        for l, v in enumerate(per_layer_vecs):
            for b in range(num_bins):
                s = b * self.basket_size
                e = min((b + 1) * self.basket_size, L_eff)
                if e <= s:
                    continue
                seg = v[s:e]
                mass = float(seg.sum())
                assert self.sum_abs_mass is not None
                self.sum_abs_mass[l, b] += mass
                self.sum_abs[l, b] += float(seg.mean())
                self.count_abs[l, b] += 1.0

    def _accumulate_baskets_relative(self, per_layer_vecs: List[torch.Tensor]) -> None:
        # per_layer_vecs: list of [L_eff]
        L_eff = per_layer_vecs[0].shape[0]
        for v in per_layer_vecs:
            assert v.shape == (L_eff,)
        assert L_eff >= self.rel_bins

        if self.num_rel_bins is None:
            self.num_rel_bins = self.rel_bins
        else:
            assert self.num_rel_bins == self.rel_bins
        self._ensure_rel_buffers(self.num_rel_bins)

        edges = torch.linspace(0, L_eff, steps=self.rel_bins + 1)
        edges = torch.floor(edges).to(torch.int64)
        edges[-1] = L_eff
        for k in range(1, edges.numel()):
            if edges[k] <= edges[k - 1]:
                edges[k] = min(edges[k - 1] + 1, L_eff)

        for l, v in enumerate(per_layer_vecs):
            for b in range(self.rel_bins):
                s = int(edges[b].item())
                e = int(edges[b + 1].item())
                if e <= s:
                    continue
                seg = v[s:e]
                mass = float(seg.sum())
                assert self.sum_rel_mass is not None
                self.sum_rel_mass[l, b] += mass
                self.sum_rel[l, b] += float(seg.mean())
                self.count_rel[l, b] += 1.0

    def _accumulate_articles(
        self, per_layer_vecs: List[torch.Tensor], spans: List[Tuple[int, int]]
    ) -> None:
        num_articles = len(spans)
        if self.num_articles is None:
            self.num_articles = num_articles
        else:
            assert self.num_articles == num_articles

        self._ensure_abs_buffers(num_articles)

        for l, v in enumerate(per_layer_vecs):
            for a_idx, (s, e) in enumerate(spans):
                # Skip empty spans (possible when excluding the first token)
                if not (0 <= s < e <= v.shape[0]):
                    continue
                seg = v[s:e]
                mass = float(seg.sum())
                assert self.sum_abs_mass is not None
                self.sum_abs_mass[l, a_idx] += mass
                self.sum_abs[l, a_idx] += float(seg.mean())
                self.count_abs[l, a_idx] += 1.0

    def _accumulate_maps_absolute(self, M: torch.Tensor) -> None:
        # M: [L_eff, L_eff], averaged over heads and layers
        assert M.dim() == 2 and M.shape[0] == M.shape[1]
        L_eff = M.shape[0]
        num_bins = (L_eff + self.basket_size - 1) // self.basket_size
        self._ensure_map_abs_buffers(num_bins)
        for bs in range(num_bins):
            rs = bs * self.basket_size
            re = min((bs + 1) * self.basket_size, L_eff)
            if re <= rs:
                continue
            for bt in range(num_bins):
                cs = bt * self.basket_size
                ce = min((bt + 1) * self.basket_size, L_eff)
                if ce <= cs:
                    continue
                block = M[rs:re, cs:ce]
                val = float(block.mean())
                self.map_abs_sum[bs, bt] += val
                self.map_abs_count[bs, bt] += 1.0

    def _accumulate_maps_relative(self, M: torch.Tensor) -> None:
        # M: [L_eff, L_eff], averaged over heads and layers
        assert M.dim() == 2 and M.shape[0] == M.shape[1]
        L_eff = M.shape[0]
        assert L_eff >= self.rel_bins
        self._ensure_map_rel_buffers()
        edges = torch.linspace(0, L_eff, steps=self.rel_bins + 1)
        edges = torch.floor(edges).to(torch.int64)
        edges[-1] = L_eff
        for k in range(1, edges.numel()):
            if edges[k] <= edges[k - 1]:
                edges[k] = min(edges[k - 1] + 1, L_eff)
        for bs in range(self.rel_bins):
            rs = int(edges[bs].item())
            re = int(edges[bs + 1].item())
            if re <= rs:
                continue
            for bt in range(self.rel_bins):
                cs = int(edges[bt].item())
                ce = int(edges[bt + 1].item())
                if ce <= cs:
                    continue
                block = M[rs:re, cs:ce]
                val = float(block.mean())
                self.map_rel_sum[bs, bt] += val
                self.map_rel_count[bs, bt] += 1.0

    def _accumulate_maps_absolute_stream(
        self,
        attentions: List[torch.Tensor],
        i: int,
        eff_idx: torch.Tensor,
        num_layers: int,
        num_heads: int,
        abs_bins: List[Tuple[int, int]],
    ) -> None:
        # Two-stage reduction: sum over heads and rows once per layer, reuse across all col bins.
        L_eff = eff_idx.numel()
        self._ensure_map_abs_buffers(len(abs_bins))

        device0 = attentions[0].device  # model device
        dtype0 = attentions[0].dtype

        for bs, (rs, re) in enumerate(abs_bins):
            rows = eff_idx[rs:re]
            r = re - rs
            assert r > 0
            # For each column bin, accumulate sums across layers on GPU, one sync per bin
            # Precompute per-layer s_cols_eff for this row-bin
            per_layer_scols: List[torch.Tensor] = []
            per_layer_scols_reserve = []
            for l in range(num_layers):
                a = attentions[l][i]  # [H, L, L]
                # Sum over heads and rows -> [L]
                temp = a[:, rows, :]  # [H, r, L]
                s_cols_full = temp.sum(dim=(0, 1))  # [L]
                # Bring into reduced (effective) index space
                s_cols_eff = torch.index_select(s_cols_full, 0, eff_idx)  # [L_eff]
                per_layer_scols.append(s_cols_eff)
                del temp, s_cols_full, a

            for bt, (cs, ce) in enumerate(abs_bins):
                c = ce - cs
                if c <= 0:
                    continue
                # GPU accumulator scalar
                acc_sum = torch.zeros((), device=device0, dtype=dtype0)
                for l in range(num_layers):
                    s_cols_eff = per_layer_scols[l]  # [L_eff]
                    block_sum = s_cols_eff[cs:ce].sum()  # scalar on GPU
                    acc_sum = acc_sum + block_sum
                denom = float(num_layers * num_heads * r * c)
                val = (acc_sum / denom).item()  # one sync per bin pair
                self.map_abs_sum[bs, bt] += val
                self.map_abs_count[bs, bt] += 1.0

            # free per-layer cache for this row-bin
            for t in per_layer_scols:
                del t

    def _accumulate_maps_relative_stream(
        self,
        attentions: List[torch.Tensor],
        i: int,
        eff_idx: torch.Tensor,
        num_layers: int,
        num_heads: int,
        rel_bins: List[Tuple[int, int]],
    ) -> None:
        # Two-stage reduction with precomputed relative bins
        L_eff = eff_idx.numel()
        assert L_eff >= self.rel_bins
        self._ensure_map_rel_buffers()

        device0 = attentions[0].device
        dtype0 = attentions[0].dtype

        for bs, (rs, re) in enumerate(rel_bins):
            rows = eff_idx[rs:re]
            r = re - rs
            if r <= 0:
                continue
            # Cache per-layer sums across heads and rows
            per_layer_scols: List[torch.Tensor] = []
            for l in range(num_layers):
                a = attentions[l][i]  # [H, L, L]
                temp = a[:, rows, :]  # [H, r, L]
                s_cols_full = temp.sum(dim=(0, 1))  # [L]
                s_cols_eff = torch.index_select(s_cols_full, 0, eff_idx)  # [L_eff]
                per_layer_scols.append(s_cols_eff)
                del temp, s_cols_full, a

            for bt, (cs, ce) in enumerate(rel_bins):
                c = ce - cs
                if c <= 0:
                    continue
                acc_sum = torch.zeros((), device=device0, dtype=dtype0)
                for l in range(num_layers):
                    s_cols_eff = per_layer_scols[l]
                    block_sum = s_cols_eff[cs:ce].sum()
                    acc_sum = acc_sum + block_sum
                denom = float(num_layers * num_heads * r * c)
                val = (acc_sum / denom).item()
                self.map_rel_sum[bs, bt] += val
                self.map_rel_count[bs, bt] += 1.0

            for t in per_layer_scols:
                del t

    def finalize(self) -> Dict[str, Any]:
        assert self.num_layers is not None
        result: Dict[str, Any] = {
            "analysis_mode": self.analysis_mode,
            "exclude_first_token": self.exclude_first_token,
            "exclude_last_token": self.exclude_last_token,
            "only_from_first_token": self.only_from_first_token,
            "compute_maps": self.compute_maps,
            "compute_incoming": self.compute_incoming,
            "num_layers": self.num_layers,
            "examples_seen": self.examples_seen,
        }

        # Attach first/last token aggregates if available
        if self.sum_first is not None and self.sum_last is not None:
            first_means = (self.sum_first / self.count_first.clamp_min(1.0)).tolist()
            last_means = (self.sum_last / self.count_last.clamp_min(1.0)).tolist()
            result.update(
                {
                    "first_token": {
                        "per_layer_means": first_means,
                        "counts": self.count_first.tolist(),
                    },
                    "last_token": {
                        "per_layer_means": last_means,
                        "counts": self.count_last.tolist(),
                    },
                }
            )

        if self.analysis_mode == "articles":
            # Only emit article aggregates when incoming vectors were computed
            articles_section = None
            if (
                self.compute_incoming
                and self.sum_abs is not None
                and self.count_abs is not None
            ):
                assert self.sum_abs_mass is not None
                denom_abs = self.count_abs.clamp_min(1.0)
                means_abs = (self.sum_abs / denom_abs).tolist()
                mass_means_abs = (self.sum_abs_mass / denom_abs).tolist()
                articles_section = {
                    "num_articles": self.sum_abs.shape[1],
                    "per_layer_article_means": means_abs,
                    "per_layer_article_mass_means": mass_means_abs,
                    "counts": self.count_abs.tolist(),
                }
            result.update({"articles": articles_section})
        else:
            # Baskets mode: build sections conditionally
            baskets_absolute = None
            baskets_relative = None
            if (
                self.compute_incoming
                and self.sum_abs is not None
                and self.count_abs is not None
            ):
                assert self.sum_abs_mass is not None
                denom_abs = self.count_abs.clamp_min(1.0)
                means_abs = (self.sum_abs / denom_abs).tolist()
                mass_means_abs = (self.sum_abs_mass / denom_abs).tolist()
                baskets_absolute = {
                    "basket_size": self.basket_size,
                    "num_bins": self.sum_abs.shape[1],
                    "per_layer_bin_means": means_abs,
                    "per_layer_bin_mass_means": mass_means_abs,
                    "counts": self.count_abs.tolist(),
                }
            # if (
            #     self.compute_incoming
            #     and self.sum_rel is not None
            #     and self.count_rel is not None
            # ):
            #     means_rel = (self.sum_rel / self.count_rel.clamp_min(1.0)).tolist()
            #     baskets_relative = {
            #         "num_bins": self.sum_rel.shape[1],
            #         "per_layer_bin_means": means_rel,
            #         "counts": self.count_rel.tolist(),
            #     }

            # Compute attention maps means when present
            maps_abs = None
            maps_abs_counts = None
            if (
                self.compute_maps
                and self.map_abs_sum is not None
                and self.map_abs_count is not None
            ):
                maps_abs = (
                    self.map_abs_sum / self.map_abs_count.clamp_min(1.0)
                ).tolist()
                maps_abs_counts = self.map_abs_count.tolist()
            maps_rel = None
            maps_rel_counts = None
            if (
                self.compute_maps
                and self.map_rel_sum is not None
                and self.map_rel_count is not None
            ):
                maps_rel = (
                    self.map_rel_sum / self.map_rel_count.clamp_min(1.0)
                ).tolist()
                maps_rel_counts = self.map_rel_count.tolist()

            layer_matrix = None
            if self.layer_bin_mass_sum is not None and self.layer_bin_count is not None:
                denom_layer = self.layer_bin_count.clamp_min(1.0)
                mass_means_layer = (self.layer_bin_mass_sum / denom_layer).tolist()
                num_bins_layer = self.layer_bin_mass_sum.shape[1]
                assert num_bins_layer >= 2
                middle_bins = max(0, num_bins_layer - 2)
                bin_labels = (
                    ["first_token"]
                    + [f"basket_{i}" for i in range(middle_bins)]
                    + ["last_token"]
                )
                layer_matrix = {
                    "basket_size": self.basket_size,
                    "num_bins": num_bins_layer,
                    "per_layer_bin_pair_mass_means": mass_means_layer,
                    "counts": self.layer_bin_count.tolist(),
                    "bin_labels": bin_labels,
                }

            result.update(
                {
                    "baskets_absolute": baskets_absolute,
                    "baskets_relative": baskets_relative,
                    "maps_absolute": (
                        {
                            "basket_size": self.basket_size,
                            "num_bins": None if maps_abs is None else len(maps_abs),
                            "per_bin_pair_means": maps_abs,
                            "counts": maps_abs_counts,
                        }
                        if maps_abs is not None
                        else None
                    ),
                    "maps_relative": (
                        {
                            "num_bins": None if maps_rel is None else len(maps_rel),
                            "per_bin_pair_means": maps_rel,
                            "counts": maps_rel_counts,
                        }
                        if maps_rel is not None
                        else None
                    ),
                    "layer_attention_matrix": layer_matrix,
                }
            )
        return result


def analyze_from_config(args: AnalysisConfig) -> Dict[str, Any]:
    base_config = _load_json(args.config_path)

    # Optional CLI override for concat dataloader batch size
    if args.batch_size is not None:
        assert isinstance(args.batch_size, int) and args.batch_size >= 1
        base_config["batch_size_concat"] = int(args.batch_size)

    concat_loader, meta = build_concat_loader_from_config(base_config)

    device = args.device or base_config.get("device")
    analyzer = AttentionAggregator(
        model_name=meta["model_name"],
        device=device,
        analysis_mode=args.analysis_mode,
        basket_size=args.basket_size,
        rel_bins=args.rel_bins,
        exclude_first_token=args.exclude_first_token,
        exclude_last_token=args.exclude_last_token,
        only_from_first_token=args.only_from_first_token,
        compute_maps=args.compute_maps,
        compute_incoming=args.compute_incoming,
    )

    max_examples = args.max_examples
    seen = 0
    for batch in tqdm(concat_loader, desc="Processing batches"):
        analyzer.process_batch(batch)
        seen += batch["input_ids"].shape[0]
        if max_examples is not None and seen >= max_examples:
            break

    result = analyzer.finalize()

    # Compose output path
    out_base = base_config.get("embeddings_output_dir", "results")
    model_tag = base_config["model_name"].replace("/", "_")
    dataset_name = os.path.basename(
        os.path.dirname(base_config["tokenized_dataset_path"])
    )
    mode_tag = (
        f"baskets_bs{args.basket_size}_rb{args.rel_bins}"
        if args.analysis_mode == "baskets"
        else "articles"
    )
    excl_first_tag = "exclude-first" if args.exclude_first_token else "include-first"
    excl_last_tag = (
        "exclude-last" if getattr(args, "exclude_last_token", False) else "include-last"
    )
    src_tag = "from-first-only" if args.only_from_first_token else "from-all"
    source_lang = base_config.get("source_lang", "en")
    target_lang = base_config.get("target_lang", None)
    indices_name = os.path.basename(base_config["indices_path"])
    out_dir = os.path.join(
        out_base,
        "_attention",
        f"{model_tag}__{indices_name}__{source_lang}__{target_lang}",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"attn__{mode_tag}__{excl_first_tag}__{excl_last_tag}__{src_tag}.json",
    )

    payload = {
        "meta": meta,
        "analysis": result,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    return {"output_path": out_path, "summary": result}


def _parse_args() -> AnalysisConfig:
    p = argparse.ArgumentParser(description="Attention analyzer for concatenated docs")
    p.add_argument(
        "--config",
        required=True,
        help="Path to config JSON used to build concat dataset",
    )
    p.add_argument(
        "--analysis_mode",
        choices=["baskets", "articles"],
        default="baskets",
        help="Aggregate attention by baskets or by article boundaries",
    )
    p.add_argument(
        "--basket_size",
        type=int,
        default=128,
        help="Basket size in tokens (baskets mode)",
    )
    p.add_argument(
        "--rel_bins",
        type=int,
        default=20,
        help="Number of relative (percentile) bins in baskets mode",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override concat dataloader batch size (default reads from config)",
    )
    p.add_argument(
        "--exclude_first_token",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude the first token (e.g., CLS/BOS) from target positions (default: exclude)",
    )
    p.add_argument(
        "--exclude_last_token",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude the last token (e.g., SEP/EOS) from target positions (default: exclude)",
    )
    p.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Process at most this many examples",
    )
    p.add_argument(
        "--device", type=str, default=None, help="Device override: cpu|cuda|mps"
    )
    p.add_argument(
        "--only_from_first_token",
        action="store_true",
        help="If set, compute contributions using only the first token as source (CLS/<s>)",
    )
    p.add_argument(
        "--exclude_maps",
        action="store_true",
        help="Exclude attention map computation to reduce VRAM and runtime (baskets mode only)",
    )
    p.add_argument(
        "--exclude_incoming",
        action="store_true",
        help="Exclude incoming attention vector computations; only compute attention maps (baskets mode)",
    )
    a = p.parse_args()
    return AnalysisConfig(
        config_path=a.config,
        analysis_mode=a.analysis_mode,
        basket_size=a.basket_size,
        rel_bins=a.rel_bins,
        exclude_first_token=bool(a.exclude_first_token),
        exclude_last_token=bool(a.exclude_last_token),
        max_examples=a.max_examples,
        batch_size=a.batch_size,
        device=a.device,
        only_from_first_token=bool(a.only_from_first_token),
        compute_maps=not bool(a.exclude_maps),
        compute_incoming=not bool(a.exclude_incoming),
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    args = _parse_args()

    out = analyze_from_config(args)


if __name__ == "__main__":
    main()
