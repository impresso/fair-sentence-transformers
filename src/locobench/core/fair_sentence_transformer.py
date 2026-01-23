import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import List, Dict, Union, Literal, Optional, Any, Tuple, Callable, Set
import tqdm
import nnsight
import numpy as np

import warnings
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling


class FairSentenceTransformer(SentenceTransformer):
    """SentenceTransformers-compatible embedder with positional fairness calibration.

    This class extends SentenceTransformer to provide attention calibration that mitigates
    positional bias in transformer-based text embeddings. The calibration works by grouping
    tokens into "baskets" of equal size and redistributing attention weights so that each
    basket receives equal total attention, regardless of position in the sequence.

    The calibration is applied during inference via nnsight hooks that intercept and modify
    attention weights after the softmax in specified layers.

    Args:
        model_name_or_path: HuggingFace model identifier or local path to the model.
        device: Device to run the model on (e.g., "cuda", "cpu", "mps"). If None, auto-detected.
        device_map: Device map for multi-GPU or offloading (e.g., "auto"). Mutually exclusive
            with device for placement logic.
        attention_path_override: Custom callable to resolve the attention softmax tensor
            for a given layer index. Signature: (nnsight_model, layer_idx) -> attention_tensor.
            If None, uses predefined paths from ATTENTION_PATHS.
        pooling_override: Force a specific pooling strategy instead of inferring from the model.
            One of "cls" (first token), "mean" (average of all tokens), or "last" (last token).
        calibrated_tokens_override: Force which token positions receive calibrated attention.
            One of "cls", "all", or "last". Defaults to matching the pooling strategy.
        padding_side_override: Force padding side instead of inferring from architecture.
            One of "left" or "right".
        trust_remote_code: Whether to trust remote code when loading the model.
        **kwargs: Additional arguments passed to SentenceTransformer.
    """

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
        padding_side_override: Optional[Literal["left", "right"]] = None,
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
        self._padding_side_override = padding_side_override
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
        override = self._padding_side_override
        if override is not None:
            assert override in ("left", "right")
            return override
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
        isolate_bos: bool,
        isolate_eos: bool,
        bos_weight: Union[float, Literal["equal"]],
        eos_weight: Union[float, Literal["equal"]],
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

        n_isolated = int(isolate_bos) + int(isolate_eos)
        content_len = valid_len - n_isolated
        assert torch.all(
            content_len > 0
        ), "content_len must be > 0 after isolating BOS/EOS"

        source_mode = self.calib_source_mode
        if source_mode == "cls":
            pool_pos = start_idx
            q_mask = torch.zeros((B, Q), device=attn.device, dtype=torch.bool)
            q_mask[torch.arange(B, device=attn.device), pool_pos] = True
        elif source_mode == "all":
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
        end_idx_b = (start_idx + valid_len - 1).view(B, 1, 1, 1)
        bos_pos_b = start_idx_b
        eos_pos_b = end_idx_b

        is_bos = pos == bos_pos_b
        is_eos = pos == eos_pos_b

        content_offset = int(isolate_bos)
        if isolate_bos:
            content_start_b = start_idx_b + 1
        else:
            content_start_b = start_idx_b
        content_rel = (pos - content_start_b).clamp_min(0)

        n_content_baskets = ((content_len + S - 1) // S).to(dtype=torch.long)
        max_content_baskets = int(n_content_baskets.max().item())
        n_total_baskets = n_content_baskets + n_isolated
        max_total_baskets = max_content_baskets + n_isolated

        content_basket_ids = content_offset + (content_rel // S)
        content_basket_ids = content_basket_ids.clamp_max(
            content_offset + max_content_baskets - 1
        )

        basket_ids = content_basket_ids.clone()
        if isolate_bos:
            basket_ids = torch.where(is_bos, torch.zeros_like(basket_ids), basket_ids)
        if isolate_eos:
            eos_basket_id = (content_offset + n_content_baskets.view(B, 1, 1, 1)).to(
                basket_ids.dtype
            )
            basket_ids = torch.where(is_eos, eos_basket_id, basket_ids)

        basket_ids = basket_ids.expand(B, H, Q, K)
        assert basket_ids.shape == (B, H, Q, K)

        mask_bk = valid_mask.view(B, 1, 1, K).to(dtype=attn.dtype)
        attn_masked = attn * mask_bk

        is_content = (
            valid_mask.view(B, 1, 1, K)
            & ~(is_bos if isolate_bos else torch.zeros_like(is_bos))
            & ~(is_eos if isolate_eos else torch.zeros_like(is_eos))
        )
        is_content = is_content.to(dtype=attn.dtype)

        bucket_sums = torch.zeros(
            (B, H, Q, max_total_baskets), device=attn.device, dtype=attn.dtype
        )
        content_attn = attn_masked * is_content
        bucket_sums = bucket_sums.scatter_add(
            dim=-1, index=basket_ids.to(torch.long), src=content_attn
        )
        denom = bucket_sums.gather(dim=-1, index=basket_ids.to(torch.long))

        n_content_baskets_f = n_content_baskets.to(dtype=attn.dtype).view(B, 1, 1, 1)
        n_total_baskets_f = n_total_baskets.to(dtype=attn.dtype).view(B, 1, 1, 1)

        calibrated = torch.where(
            denom > 0,
            (content_attn / denom) * (1.0 / n_total_baskets_f),
            torch.zeros_like(content_attn),
        )

        bos_attn_value = (
            (attn_masked * is_bos.to(attn.dtype)).sum(dim=-1, keepdim=True)
            if isolate_bos
            else None
        )
        eos_attn_value = (
            (attn_masked * is_eos.to(attn.dtype)).sum(dim=-1, keepdim=True)
            if isolate_eos
            else None
        )

        if isolate_bos:
            if bos_weight == "equal":
                calibrated = calibrated + torch.where(
                    is_bos,
                    torch.ones_like(attn_masked) / n_total_baskets_f,
                    torch.zeros_like(attn_masked),
                )
            else:
                assert isinstance(bos_weight, (int, float)) and 0.0 <= bos_weight <= 1.0
                calibrated = calibrated + bos_weight * attn_masked * is_bos.to(
                    attn.dtype
                )

        if isolate_eos:
            if eos_weight == "equal":
                calibrated = calibrated + torch.where(
                    is_eos,
                    torch.ones_like(attn_masked) / n_total_baskets_f,
                    torch.zeros_like(attn_masked),
                )
            else:
                assert isinstance(eos_weight, (int, float)) and 0.0 <= eos_weight <= 1.0
                calibrated = calibrated + eos_weight * attn_masked * is_eos.to(
                    attn.dtype
                )

        bos_uses_float = isolate_bos and isinstance(bos_weight, (int, float))
        eos_uses_float = isolate_eos and isinstance(eos_weight, (int, float))
        if bos_uses_float or eos_uses_float:
            bos_final = (
                float(bos_weight) * bos_attn_value
                if bos_uses_float
                else bos_attn_value / n_total_baskets_f
            )
            eos_final = (
                float(eos_weight) * eos_attn_value
                if eos_uses_float
                else eos_attn_value / n_total_baskets_f
            )
            remaining_for_content = 1.0 - bos_final - eos_final
            current_content_sum = (
                calibrated.sum(dim=-1, keepdim=True) - bos_final - eos_final
            )
            content_scale = torch.where(
                current_content_sum > 0,
                remaining_for_content / current_content_sum,
                torch.ones_like(current_content_sum),
            )
            calibrated = torch.where(
                is_content.bool(),
                calibrated * content_scale,
                calibrated,
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

        if strength == 1.0:
            return normalized
        return torch.where(
            q_mask_b,
            normalized * strength + attn_masked * (1.0 - strength),
            attn_masked,
        )

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

    def _resolve_isolation_defaults(
        self,
        isolate_bos: Optional[bool],
        isolate_eos: Optional[bool],
    ) -> Tuple[bool, bool]:
        if self.pooling_strategy == "mean":
            default_bos, default_eos = False, False
        elif self.pooling_strategy == "cls":
            default_bos, default_eos = True, False
        elif self.pooling_strategy == "last":
            default_bos, default_eos = True, True
        else:
            default_bos, default_eos = True, True
        resolved_bos = default_bos if isolate_bos is None else isolate_bos
        resolved_eos = default_eos if isolate_eos is None else isolate_eos
        return resolved_bos, resolved_eos

    def encode_positionally_fair(
        self,
        sentences: Union[str, List[str]],
        *,
        calib_basket_size: int,
        calib_layers: int,
        calib_strength: float,
        isolate_bos: Optional[bool] = None,
        isolate_eos: Optional[bool] = None,
        bos_weight: Union[float, Literal["equal"]] = "equal",
        eos_weight: Union[float, Literal["equal"]] = "equal",
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        device: Optional[str] = None,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Encode sentences with positional fairness calibration applied to attention weights.

        Tokens (excluding optionally isolated BOS/EOS) are grouped into consecutive baskets
        of size `calib_basket_size`. Attention weights are redistributed so each basket
        receives equal total attention (1/n_baskets), eliminating positional bias where
        early or late tokens dominate the embedding.

        Sentences shorter than `calib_basket_size` (after accounting for isolated tokens)
        fall back to standard `.encode()` without calibration.

        Args:
            sentences: Single sentence string or list of sentences to encode.
            calib_basket_size: Number of consecutive content tokens per basket. Larger values
                create coarser calibration; smaller values provide finer-grained fairness.
            calib_layers: Number of final transformer layers to apply calibration to.
                E.g., calib_layers=3 calibrates the last 3 layers.
            calib_strength: Interpolation factor between calibrated (1.0) and original (0.0)
                attention. Value in [0, 1].
            isolate_bos: Whether to treat BOS token as its own basket. If None, inferred
                from pooling strategy (True for cls/last pooling, False for mean).
            isolate_eos: Whether to treat EOS token as its own basket. If None, inferred
                from pooling strategy (True for last pooling, False otherwise).
            bos_weight: Weight for BOS token attention. Either "equal" (same as content
                baskets, i.e., 1/n_baskets) or a float in [0, 1] for fixed proportion.
            eos_weight: Weight for EOS token attention. Either "equal" or float in [0, 1].
            batch_size: Number of sentences to process simultaneously.
            show_progress_bar: Whether to display a tqdm progress bar during encoding.
            normalize_embeddings: Whether to L2-normalize the output embeddings.
            device: Device to run inference on. If None, uses the model's default device.
            convert_to_numpy: Return embeddings as numpy array (default True).
            convert_to_tensor: Return embeddings as torch.Tensor. Takes precedence over
                convert_to_numpy if both are True.

        Returns:
            Embeddings with shape (n_sentences, embedding_dim) as numpy array or torch.Tensor.
        """
        assert calib_basket_size > 0
        assert calib_layers > 0
        assert 0.0 <= calib_strength <= 1.0

        resolved_bos, resolved_eos = self._resolve_isolation_defaults(
            isolate_bos, isolate_eos
        )

        if isinstance(bos_weight, (int, float)):
            assert 0.0 <= bos_weight <= 1.0, "bos_weight must be in [0, 1]"
        else:
            assert (
                bos_weight == "equal"
            ), f"bos_weight must be float or 'equal', got {bos_weight}"
        if isinstance(eos_weight, (int, float)):
            assert 0.0 <= eos_weight <= 1.0, "eos_weight must be in [0, 1]"
        else:
            assert (
                eos_weight == "equal"
            ), f"eos_weight must be float or 'equal', got {eos_weight}"

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

        n_isolated = int(resolved_bos) + int(resolved_eos)
        preflight_enc = self.tokenizer(
            sentences_list, padding=False, truncation=True, return_tensors=None
        )
        content_lens = [len(ids) - n_isolated for ids in preflight_enc["input_ids"]]
        short_idxs = [i for i, cl in enumerate(content_lens) if cl < calib_basket_size]
        calib_idxs = [i for i, cl in enumerate(content_lens) if cl >= calib_basket_size]

        if short_idxs:
            print(
                f"[Info encode_positionally_fair:] {len(short_idxs)} sample(s) have content "
                f"length < basket size ({calib_basket_size}), using .encode() fallback"
            )

        if not calib_idxs:
            return self.encode(
                sentences_list,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
                device=device,
                convert_to_numpy=convert_to_numpy,
                convert_to_tensor=convert_to_tensor,
            )

        calib_sentences = [sentences_list[i] for i in calib_idxs]

        all_embeddings: List[torch.Tensor] = []
        rng = range(0, len(calib_sentences), batch_size)
        iterator = tqdm.tqdm(rng, desc="Encoding (fair)", disable=not show_progress_bar)
        for start in iterator:
            end = min(start + batch_size, len(calib_sentences))
            chunk = calib_sentences[start:end]
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

            with torch.no_grad():
                with self._nnsight_model.trace() as tracer:
                    with tracer.invoke(
                        input_ids=input_ids, attention_mask=attention_mask
                    ):
                        layer_start = max(0, self._num_layers - calib_layers)
                        for idx in range(layer_start, self._num_layers):
                            attn = self._resolve_attention_softmax(idx)
                            calibrated = self._calibrate_attention(
                                attn,
                                attention_mask,
                                basket_size=calib_basket_size,
                                strength=calib_strength,
                                isolate_bos=resolved_bos,
                                isolate_eos=resolved_eos,
                                bos_weight=bos_weight,
                                eos_weight=eos_weight,
                            )
                            attn.copy_(calibrated)
                        model_output = self._nnsight_model.output.save()

            hidden = self._extract_last_hidden_state(model_output)
            pooled = self._pool(hidden, attention_mask)
            if normalize_embeddings:
                pooled = F.normalize(pooled, p=2, dim=1)
            all_embeddings.append(pooled.cpu())
            del model_output, hidden, pooled, input_ids, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        calib_embeddings = torch.cat(all_embeddings, dim=0)
        assert calib_embeddings.shape[0] == len(calib_idxs)

        if not short_idxs:
            embeddings = calib_embeddings
        else:
            short_sentences = [sentences_list[i] for i in short_idxs]
            short_embeddings = self.encode(
                short_sentences,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=normalize_embeddings,
                device=device,
                convert_to_numpy=False,
                convert_to_tensor=True,
            )
            assert short_embeddings.shape[0] == len(short_idxs)
            D = calib_embeddings.shape[1]
            assert short_embeddings.shape[1] == D
            embeddings = torch.empty(
                (len(sentences_list), D), dtype=calib_embeddings.dtype
            )
            for out_i, orig_i in enumerate(calib_idxs):
                embeddings[orig_i] = calib_embeddings[out_i]
            for out_i, orig_i in enumerate(short_idxs):
                embeddings[orig_i] = short_embeddings[out_i]

        assert embeddings.shape[0] == len(sentences_list)
        if convert_to_tensor:
            return embeddings
        if convert_to_numpy:
            return embeddings.detach().numpy()
        return embeddings
