"""
Merged analysis functionality for position similarities and directional leakage.
This module provides a unified approach to plotting multiple analysis results.
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple, Callable
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from collections import defaultdict
from pathlib import Path

from .plot_constants import (
    SUBPLOT_COLUMN_SPACING,
    SUBPLOT_VERTICAL_SCALE,
    SUBPLOT_FONT_SCALE,
    BASE_AXIS_LABEL_FONT_SIZE,
    BASE_TICK_LABEL_FONT_SIZE,
    BASE_SUBPLOT_TITLE_FONT_SIZE,
    LEGEND_LINEWIDTH,
    POS_LABEL_PAD,
    PLOT_LINEWIDTH,
    PLOT_MARKERSIZE,
)

# Import the required analyses functions
from ..analysis.segment_embedding_analysis import (
    DocumentSegmentSimilarityAnalyzer,
    DirectionalLeakageAnalyzer,
    PositionalDirectionalLeakageAnalyzer,
    compute_position_token_lengths,
)

from .single_plotters import (
    DirectionalLeakageSinglePlotter,
    PositionSimilaritySinglePlotter,
    PositionalDirectionalLeakageSinglePlotter,
    PositionalDirectionalLeakageHeatmapSinglePlotter,
)


class DirectionalLeakageMultiPlotter:
    """
    Class to handle the plotting of directional leakage analysis results in a multi plot.

    NOTE: The averages computed by this plotter may differ slightly from PositionalDirectionalLeakageMultiPlotter
    due to different averaging methodologies:
    - This plotter: Simple mean over all pairwise similarities
    - PositionalDirectionalLeakageMultiPlotter: Averages per-position means (equal weight per position)

    """

    def __init__(self):
        self.analysis_type = "directional_leakage"

    def plot(
        self,
        paths: List[str | Path],
        pooling_strategy_segment_standalone: str = "cls",
        show_segment_lengths: bool = False,
        figure_width: int = 15,
        subplot_height: int = 5,
        save_plot: bool = False,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        return_full_results: bool = False,
        single_model_mode: Optional[bool] = None,
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Analyze and plot directional leakage for multiple experiments in a grid,
        organized by model name and concat size.

        Args:
            paths: List of paths to experiment results
            pooling_strategy_segment_standalone: Either "cls" or "mean" for standalone segment embeddings pooling
            figure_width: Width of the complete figure in inches
            subplot_height: Height of each subplot in inches
            save_plot: Whether to save the plot
            save_path: Path to save the figure (if None and save_plot is True, saves to first path directory)
            show_plot: Whether to display the plot
            return_full_results: Whether to return the full analysis results
            single_model_mode: If True, optimize layout for single model. If None, auto-detect based on data

        Returns:
            If return_full_results is True, returns a dictionary with model names as keys and
            lists of result dictionaries as values
        """
        directional_leakage_single_plotter = DirectionalLeakageSinglePlotter()

        return analyze_and_plot_multiple_results(
            paths=paths,
            analysis_type=self.analysis_type,
            title="Directional Information Flow",
            pooling_legend_type="segment_standalone",
            subplotter=directional_leakage_single_plotter.plot_directional_leakage_in_subplot,
            pooling_strategy_segment_standalone=pooling_strategy_segment_standalone,
            show_segment_lengths=show_segment_lengths,
            figure_width=figure_width,
            subplot_height=subplot_height,
            save_plot=save_plot,
            save_path=save_path,
            show_plot=show_plot,
            return_full_results=return_full_results,
            single_model_mode=single_model_mode,
        )


class DocumentLevel2SegmentStandaloneSimPlotter:
    """
    Class to handle the plotting of position analysis results in a multi plot.
    """

    def __init__(self):
        self.analysis_type = "position"
        self.document_embedding_type = "document-level"

    def plot(
        self,
        paths: List[str | Path],
        pooling_strategy_segment_standalone: str = "cls",
        pooling_strategy_document: str = "cls",
        matryoshka_dimensions: Optional[List[int]] = None,
        show_segment_lengths: bool = False,
        show_lengths: bool = False,
        figure_width: int = 15,
        subplot_height: int = 5,
        save_plot: bool = False,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        return_full_results: bool = False,
        single_model_mode: Optional[bool] = None,
        split_plots_by_source_lang: bool = False,
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Analyze and plot position-based similarity metrics for multiple experiments in a grid,
        organized by model name and concat size.

        Args:
            paths: List of paths to experiment results
            pooling_strategy_segment_standalone: Either "cls" or "mean" for standalone segment embeddings pooling
            pooling_strategy_document: Either "cls" or "mean" for document embeddings pooling
            matryoshka_dimensions: Optional list of dimensions to truncate embeddings to for Matryoshka analysis
            show_segment_lengths: Whether to show segment length information in titles and legends (default: True)
            figure_width: Width of the complete figure in inches
            subplot_height: Height of each subplot in inches
            save_plot: Whether to save the plot
            save_path: Path to save the figure (if None and save_plot is True, saves to first path directory)
            show_plot: Whether to display the plot
            return_full_results: Whether to return the full analysis results

        Returns:
            If return_full_results is True, returns a dictionary with model names as keys and
            lists of result dictionaries as values
        """

        position_similarity_single_plotter = PositionSimilaritySinglePlotter()

        return analyze_and_plot_multiple_results(
            paths=paths,
            analysis_type=self.analysis_type,
            document_embedding_type=self.document_embedding_type,
            # title="Similarity between Document-Level Embedding and Standalone Segment Embeddings",
            title="Similarity D-Level Embedding and S-Level Embeddings",
            pooling_legend_type="segment_standalone_and_document",
            subplotter=position_similarity_single_plotter.plot_position_similarities_in_subplot,
            pooling_strategy_segment_standalone=pooling_strategy_segment_standalone,
            pooling_strategy_document=pooling_strategy_document,
            matryoshka_dimensions=matryoshka_dimensions,
            show_segment_lengths=show_segment_lengths,
            show_lengths=show_lengths,
            figure_width=figure_width,
            subplot_height=subplot_height,
            save_plot=save_plot,
            save_path=save_path,
            show_plot=show_plot,
            return_full_results=return_full_results,
            single_model_mode=single_model_mode,
        )

    def plot_multi_models(
        self,
        paths: List[str | Path],
        model_pooling_strats: Dict[str, str],
        merge_latechunk_jina: bool = False,
        show_segment_lengths: bool = False,
        show_lengths: bool = False,
        figure_width: int = 15,
        subplot_height: int = 5,
        save_plot: bool = False,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        return_full_results: bool = False,
        single_model_mode: Optional[bool] = None,
        split_plots_by_target_lang: bool = False,
        split_plots_by_source_lang: bool = False,
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Analyze and plot position-based similarity metrics for multiple models in a grid,
        organized by concat size and language, with different models shown as different lines.

        Args:
            paths: List of paths to experiment results
            model_pooling_strats: Dictionary mapping model names to their pooling strategies
                                 (e.g., {"Alibaba-NLP/gte-multilingual-base": "cls",
                                        "jinaai/jina-embeddings-v3": "mean"})
            show_segment_lengths: Whether to show segment length information in titles and legends
            show_lengths: Whether to show token lengths as bar charts on right y-axis
            figure_width: Width of the complete figure in inches
            subplot_height: Height of each subplot in inches
            save_plot: Whether to save the plot
            save_path: Path to save the figure (if None and save_plot is True, saves to first path directory)
            show_plot: Whether to display the plot
            return_full_results: Whether to return the full analysis results
            single_model_mode: If True, optimize layout for single model. If None, auto-detect based on data

        Returns:
            If return_full_results is True, returns a dictionary with model names as keys and
            lists of result dictionaries as values
        """
        # Sanity: both split flags cannot be true at the same time
        if split_plots_by_target_lang and split_plots_by_source_lang:
            raise ValueError(
                "Only one of split_plots_by_target_lang or split_plots_by_source_lang may be True."
            )

        # Create a custom single plotter that shows different models instead of matryoshka dimensions
        multi_model_single_plotter = MultiModelPositionSimilaritySinglePlotter(
            model_pooling_strats, paths
        )

        # If not splitting by target language, keep existing behavior
        if not split_plots_by_target_lang and not split_plots_by_source_lang:
            return self._analyze_and_plot_multiple_results_multi_model(
                paths=paths,
                model_pooling_strats=model_pooling_strats,
                analysis_type=self.analysis_type,
                document_embedding_type=self.document_embedding_type,
                # title="Similarity between Document-Level Embedding and Standalone Segment Embeddings",
                title="Similarity D-Level Embedding and S-Level Embeddings",
                pooling_legend_type="segment_standalone_and_document",
                subplotter=multi_model_single_plotter,  # Pass the object, not the method
                pooling_strategy_segment_standalone="cls",  # Will be overridden by model-specific strategies
                pooling_strategy_document="cls",  # Will be overridden by model-specific strategies
                matryoshka_dimensions=None,  # We don't use matryoshka for multi-model plot
                merge_latechunk_jina=merge_latechunk_jina,
                show_segment_lengths=show_segment_lengths,
                show_lengths=show_lengths,
                figure_width=figure_width,
                subplot_height=subplot_height,
                save_plot=save_plot,
                save_path=save_path,
                show_plot=show_plot,
                return_full_results=return_full_results,
                single_model_mode=single_model_mode,
                global_ylim_override=(0.28, 1.0),
            )

        # Split the inputs and ensure identical formatting across all figures
        from ..analysis.segment_embedding_analysis import load_exp_info

        # Build grouping map based on requested split dimension
        groups: Dict[str, List[str | Path]] = defaultdict(list)
        path_to_model: Dict[str | Path, str] = {}
        for p in paths:
            exp = load_exp_info(p)
            if split_plots_by_target_lang:
                key = (
                    exp.get("target_lang")
                    if exp.get("target_lang") is not None
                    else "mono"
                )
            else:
                key = exp.get("source_lang", "unknown")
            groups[str(key)].append(p)
            path_to_model[p] = exp["model_name"]

        # Pre-compute a global ylim across ALL paths to guarantee identical y-axis across figures
        doc_seg_analyzer = DocumentSegmentSimilarityAnalyzer()
        all_y_values: List[float] = []
        for p in paths:
            model_name = path_to_model[p]
            pooling = model_pooling_strats.get(model_name, "cls")
            res = doc_seg_analyzer.run_position_analysis(
                path=p,
                document_embedding_type=self.document_embedding_type,
                pooling_strategy_segment_standalone=pooling,
                pooling_strategy_document=pooling,
                matryoshka_dimensions=None,
            )
            all_y_values.extend(res["position_means"])
            all_y_values.extend(res["position_ci_lower"])
            all_y_values.extend(res["position_ci_upper"])

            if merge_latechunk_jina and model_name == "jinaai/jina-embeddings-v3":
                latechunk_res = doc_seg_analyzer.run_position_analysis(
                    path=p,
                    document_embedding_type="latechunk-segment",
                    pooling_strategy_segment_standalone=pooling,
                    pooling_strategy_document="cls",
                    matryoshka_dimensions=None,
                )
                all_y_values.extend(latechunk_res["position_means"])
                all_y_values.extend(latechunk_res["position_ci_lower"])
                all_y_values.extend(latechunk_res["position_ci_upper"])

        global_ylim: Optional[Tuple[float, float]] = (0.28, 1.0)

        # Pre-compute a global token length ylim if token lengths are shown
        global_token_ylim: Optional[Tuple[float, float]] = None
        if show_lengths:
            token_len_results = compute_position_token_lengths(paths)
            token_vals: List[float] = []
            for v in token_len_results.values():
                token_vals.extend(v.get("position_means", []))
            if token_vals:
                tmax = max(token_vals)
                tmargin = tmax * 0.05
                global_token_ylim = (0, tmax + tmargin)

        # Run one figure per group using the same ylim/token_ylim
        merged_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for grp_key, grp_paths in groups.items():
            # Build per-figure save path when requested
            per_save_path = None
            if save_plot and save_path is not None:
                base, ext = os.path.splitext(str(save_path))
                suffix = (
                    f"_tgt-{grp_key}"
                    if split_plots_by_target_lang
                    else f"_src-{grp_key}"
                )
                per_save_path = f"{base}{suffix}{ext}"

            out = self._analyze_and_plot_multiple_results_multi_model(
                paths=grp_paths,
                model_pooling_strats=model_pooling_strats,
                analysis_type=self.analysis_type,
                document_embedding_type=self.document_embedding_type,
                # title="Similarity between Document-Level Embedding and Standalone Segment Embeddings",
                title="Similarity D-Level Embedding and S-Level Embeddings",
                pooling_legend_type="segment_standalone_and_document",
                subplotter=multi_model_single_plotter,
                pooling_strategy_segment_standalone="cls",
                pooling_strategy_document="cls",
                matryoshka_dimensions=None,
                merge_latechunk_jina=merge_latechunk_jina,
                show_segment_lengths=show_segment_lengths,
                show_lengths=show_lengths,
                figure_width=figure_width,
                subplot_height=subplot_height,
                save_plot=save_plot,
                save_path=per_save_path,
                show_plot=show_plot,
                return_full_results=return_full_results,
                single_model_mode=single_model_mode,
                global_ylim_override=global_ylim,
                global_token_ylim_override=global_token_ylim,
            )

            # Merge results if requested
            if return_full_results and isinstance(out, dict) and "model_results" in out:
                for k, v in out["model_results"].items():
                    merged_results[k].extend(v)

        return {"model_results": dict(merged_results)} if return_full_results else None

    def plot_multi_calibrations(
        self,
        paths: List[str | Path],
        pooling_strategy: str = "cls",
        show_segment_lengths: bool = False,
        show_lengths: bool = False,
        figure_width: int = 15,
        subplot_height: int = 5,
        save_plot: bool = False,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        return_full_results: bool = False,
        single_model_mode: Optional[bool] = None,
        split_plots_by_target_lang: bool = False,
        split_plots_by_source_lang: bool = False,
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Similar to plot_multi_models, but shows different attention calibration settings as
        differently colored lines within each subplot for a SINGLE model (paths belong to
        the same base model, typically mGTE).

        Grouping of subplots (experiment instances) remains by (concat_size, lang pair).
        Calibration labels are derived from each path's embedding_config.json -> calibration_effective.
        """
        # Sanity: both split flags cannot be true at the same time
        if split_plots_by_target_lang and split_plots_by_source_lang:
            raise ValueError(
                "Only one of split_plots_by_target_lang or split_plots_by_source_lang may be True."
            )

        return self._analyze_and_plot_multiple_results_multi_calibration(
            paths=paths,
            pooling_strategy_segment_standalone=pooling_strategy,
            pooling_strategy_document=pooling_strategy,
            analysis_type=self.analysis_type,
            document_embedding_type=self.document_embedding_type,
            title="Similarity between Document-Level Embd. and Standalone Segment Embds. (Calibrated mGTE)",
            show_segment_lengths=show_segment_lengths,
            show_lengths=show_lengths,
            figure_width=figure_width,
            subplot_height=subplot_height,
            save_plot=save_plot,
            save_path=save_path,
            show_plot=show_plot,
            return_full_results=return_full_results,
            single_model_mode=single_model_mode,
            split_plots_by_target_lang=split_plots_by_target_lang,
            split_plots_by_source_lang=split_plots_by_source_lang,
        )

    def _analyze_and_plot_multiple_results_multi_calibration(
        self,
        paths: List[str | Path],
        analysis_type: str = "position",
        document_embedding_type: str = "document-level",
        title: str = "Similarity between Document-Level Embedding and Standalone Segment Embeddings",
        subplotter: Callable = None,
        pooling_strategy_segment_standalone: str = "cls",
        pooling_strategy_document: str = "cls",
        matryoshka_dimensions: Optional[List[int]] = None,
        show_segment_lengths: bool = False,
        show_lengths: bool = False,
        figure_width: int = 15,
        subplot_height: int = 4,
        save_plot: bool = False,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        return_full_results: bool = False,
        single_model_mode: Optional[bool] = None,
        split_plots_by_target_lang: bool = False,
        split_plots_by_source_lang: bool = False,
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Multi-plot where each subplot corresponds to one experiment instance (concat_size, lang pair),
        and each line corresponds to a different attention calibration found in the given paths.

        Reuses the grid/axis layout from analyze_and_plot_multiple_results by providing a custom
        single-plotter and suppressing the default legend so we can inject a calibration legend.
        """
        from ..analysis.segment_embedding_analysis import load_exp_info

        # --- Group paths by (concat_size, lang_key) and extract calibration labels ---
        def lang_key_from_exp(exp: Dict[str, Any]) -> str:
            src = exp.get("source_lang", "unknown")
            tgt = exp.get("target_lang")
            return src if tgt is None else f"{src}_{tgt}"

        def build_calibration_label(calib: Dict[str, Any]) -> str:
            sa = calib.get("standalone", {})
            lc = calib.get("latechunk", {})

            sa_apply = bool(sa.get("apply_attn_calibration", False))
            lc_apply = bool(lc.get("apply_attn_calibration", False))

            if sa_apply and lc_apply:
                # Require identical parameters for a unified label
                sa_tuple = (
                    str(sa.get("calib_layers")),
                    str(sa.get("calib_source_tokens")),
                    str(sa.get("calib_basket_size")),
                )
                lc_tuple = (
                    str(lc.get("calib_layers")),
                    str(lc.get("calib_source_tokens")),
                    str(lc.get("calib_basket_size")),
                )
                assert (
                    sa_tuple == lc_tuple
                ), "Calibration parameters for standalone and latechunk must match when both are applied."
                # label_string = (
                #     f"{sa_tuple[2]}\u2014{sa_tuple[1].upper()}\u2014{sa_tuple[0]}"
                # )
                label_string = f"{sa_tuple[2]}\u2014{sa_tuple[1].replace('cls', '<s>')}\u2014{sa_tuple[0]}"
                return label_string

            if (not sa_apply) and lc_apply:
                lc_tuple = (
                    str(lc.get("calib_layers")),
                    str(lc.get("calib_source_tokens")),
                    str(lc.get("calib_basket_size")),
                )
                # label_string = f"Doc_only_clb\u2014{lc_tuple[2]}\u2014{lc_tuple[1].upper()}\u2014{lc_tuple[0]}"
                label_string = f"Doc_only_clb\u2014{lc_tuple[2]}\u2014{lc_tuple[1].replace('cls', '<s>')}\u2014{lc_tuple[0]}"
                return label_string

            # If only standalone applies (rare), fail fast as undefined in spec
            assert not (
                sa_apply and not lc_apply
            ), "Standalone-only calibration not supported in this plot."
            # Neither applies
            assert False, "Neither standalone nor latechunk calibration is applied."

        # Map config_key -> list of (path, calibration_label)
        config_groups: Dict[Tuple[int, str], List[Tuple[str | Path, str]]] = (
            defaultdict(list)
        )
        config_to_model: Dict[Tuple[int, str], str] = {}
        all_labels: List[str] = []
        for p in paths:
            exp = load_exp_info(p)
            key = (exp["concat_size"], lang_key_from_exp(exp))
            # Read calibration label from embedding_config.json within path
            config_path = Path(p) / "embedding_config.json"
            with open(config_path, "r") as f:
                cfg = json.load(f)
            if "calibration_effective" in cfg:
                label = build_calibration_label(cfg["calibration_effective"])
            else:
                label = "Original mGTE (no attention calibration)"
            config_groups[key].append((p, label))
            all_labels.append(label)
            config_to_model[key] = exp["model_name"]

        # Unique labels and color mapping (consistent across figure)
        unique_labels = list(dict.fromkeys(all_labels))
        base_colors = [
            "#777DA7",
            "#49BEAA",
            "#EF767A",
            "#EEB868",
        ]
        label_colors: Dict[str, str] = {}
        for i, lab in enumerate(unique_labels):
            label_colors[lab] = base_colors[i % len(base_colors)]
        # Force a consistent color for the baseline (no calibration)
        if "Original mGTE (no attention calibration)" in label_colors:
            label_colors["Original mGTE (no attention calibration)"] = "blue"

        # Pre-compute results for all (config, path)
        doc_seg_analyzer = DocumentSegmentSimilarityAnalyzer()
        all_results: Dict[Tuple[Tuple[int, str], str], Dict[str, Any]] = {}
        all_y_values: List[float] = []
        selected_paths_for_layout: List[str | Path] = []

        for cfg_key, pl in config_groups.items():
            # pick first path as base for layout/labels
            if pl:
                selected_paths_for_layout.append(pl[0][0])
            for path, _lab in pl:
                res = doc_seg_analyzer.run_position_analysis(
                    path=path,
                    document_embedding_type=document_embedding_type,
                    pooling_strategy_segment_standalone=pooling_strategy_segment_standalone,
                    pooling_strategy_document=pooling_strategy_document,
                    matryoshka_dimensions=None,
                )
                all_results[(cfg_key, str(path))] = res
                all_y_values.extend(res["position_means"])
                all_y_values.extend(res["position_ci_lower"])
                all_y_values.extend(res["position_ci_upper"])

        # Global y-limits fixed for consistent scaling across figures
        global_ylim: Optional[Tuple[float, float]] = (0.28, 1.0)

        # Global token y-limits if required
        global_token_ylim: Optional[Tuple[float, float]] = None
        if show_lengths:
            token_len_results = compute_position_token_lengths(paths)
            token_vals: List[float] = []
            for v in token_len_results.values():
                token_vals.extend(v.get("position_means", []))
            if token_vals:
                tmax = max(token_vals)
                tmargin = tmax * 0.05
                global_token_ylim = (0, tmax + tmargin)

        # Custom subplotter that plots all calibrations for a config using precomputed results
        class _CalibrationSubplotter:
            def __init__(
                self, all_results, config_groups, label_colors, ylim, token_ylim
            ):
                self._all_results = all_results
                self._config_groups = config_groups
                self._label_colors = label_colors
                self._ylim = ylim
                self._token_ylim = token_ylim

            def plot_position_similarities_in_subplot(self, ax, base_result, **kwargs):
                # Enforce fixed limits across subplots
                kwargs["ylim"] = self._ylim
                if self._token_ylim is not None:
                    kwargs["token_ylim"] = self._token_ylim

                # Basic positions from base_result
                positions = list(range(1, len(base_result["position_means"]) + 1))

                # Identify config key of this base_result
                concat_size = base_result["concat_size"]
                src = base_result.get("source_lang", "unknown")
                tgt = base_result.get("target_lang")
                lang_key = src if tgt is None else f"{src}_{tgt}"
                cfg_key = (concat_size, lang_key)

                # Plot each calibration line
                if cfg_key in self._config_groups:
                    for path, lab in self._config_groups[cfg_key]:
                        res = self._all_results.get((cfg_key, str(path)))
                        assert res is not None, "Missing precomputed result"
                        color = self._label_colors.get(lab, "black")
                        linestyle = "--" if lab.startswith("Doc_only_clb") else "-"
                        ax.plot(
                            positions,
                            res["position_means"],
                            marker="o",
                            linestyle=linestyle,
                            color=color,
                            linewidth=PLOT_LINEWIDTH,
                            markersize=PLOT_MARKERSIZE,
                            label=lab,
                        )
                        for pos, ci_lo, ci_hi in zip(
                            positions,
                            res["position_ci_lower"],
                            res["position_ci_upper"],
                        ):
                            rect = plt.Rectangle(
                                (pos - 0.4 / 2, ci_lo),
                                0.4,
                                ci_hi - ci_lo,
                                color=color,
                                alpha=0.2,
                            )
                            ax.add_patch(rect)

                # Draw token lengths from base_result if requested
                show_lengths_local = kwargs.get("show_lengths", False)
                show_token_ylabel = kwargs.get("show_token_ylabel", True)
                show_token_ticklabels = kwargs.get("show_token_ticklabels", True)
                if show_lengths_local and "token_lengths" in base_result:
                    ax2 = ax.twinx()
                    token_means = base_result["token_lengths"]["position_means"]
                    ax2.bar(
                        positions,
                        token_means,
                        alpha=0.3,
                        color="gray",
                        width=0.6,
                        zorder=1,
                    )
                    if self._token_ylim is not None:
                        ax2.set_ylim(self._token_ylim)
                    if show_token_ylabel:
                        ax2.set_ylabel(
                            "Token Length",
                            color="gray",
                            fontsize=BASE_AXIS_LABEL_FONT_SIZE * SUBPLOT_FONT_SCALE,
                        )
                    ax2.tick_params(
                        axis="y",
                        labelcolor="gray",
                        labelright=show_token_ticklabels,
                        labelleft=False,
                        labelsize=BASE_TICK_LABEL_FONT_SIZE * SUBPLOT_FONT_SCALE,
                    )
                    if not show_token_ticklabels:
                        ax2.set_ylabel("")
                    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
                    ax.set_zorder(ax2.get_zorder() + 1)
                    ax.patch.set_visible(False)

                # Axis labels and title
                ax.set_xlabel(
                    "Position",
                    fontsize=BASE_AXIS_LABEL_FONT_SIZE * SUBPLOT_FONT_SCALE,
                    labelpad=15,
                )
                ax.set_ylabel(
                    "Cosine Similarity",
                    fontsize=BASE_AXIS_LABEL_FONT_SIZE * SUBPLOT_FONT_SCALE,
                )

                if kwargs.get("show_title", True):
                    if tgt and src:
                        title_text = f"Lang.: [{tgt}, {src}, ..., {src}]"
                    elif src:
                        title_text = f"Lang.: [{src}, ..., {src}]"
                    else:
                        title_text = f"Concat Size: {concat_size}"
                    range_id = base_result.get("range_id", "N/A")
                    if kwargs.get("show_segment_lengths", False) and range_id != "N/A":
                        title_text += f"; SL:: {range_id}"
                    ax.set_title(
                        title_text,
                        fontsize=BASE_SUBPLOT_TITLE_FONT_SIZE * SUBPLOT_FONT_SCALE,
                    )

                ax.set_xticks(positions)
                ax.tick_params(
                    axis="x",
                    labelsize=BASE_TICK_LABEL_FONT_SIZE * SUBPLOT_FONT_SCALE,
                    pad=POS_LABEL_PAD,
                )
                if self._ylim is not None:
                    ax.set_ylim((self._ylim[0], max(self._ylim[1], 1.0)))
                else:
                    bottom, top = ax.get_ylim()
                    if top < 1.0:
                        ax.set_ylim((bottom, 1.0))

                ax.set_yticks([0.4, 0.6, 0.8, 1.0])
                ax.yaxis.set_minor_locator(MultipleLocator(0.1))
                if self._ylim is not None:
                    y_range = self._ylim[1] - self._ylim[0]
                    ax.yaxis.set_major_formatter(
                        FormatStrFormatter("%.2f" if y_range < 0.3 else "%.1f")
                    )
                else:
                    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                ax.tick_params(
                    axis="y",
                    labelleft=kwargs.get("show_cosine_ticklabels", True),
                    labelsize=BASE_TICK_LABEL_FONT_SIZE * SUBPLOT_FONT_SCALE,
                )
                ax.grid(True, which="major", linestyle="--", alpha=0.7)
                ax.grid(True, which="minor", linestyle="--", alpha=0.7)

        def render_one_figure(
            group_paths: List[str | Path], per_save_path: Optional[str]
        ):
            # Build per-figure config subset and base paths
            # Columns: languages in first-seen order; Rows: concat sizes ascending
            filtered_config_groups: Dict[
                Tuple[int, str], List[Tuple[str | Path, str]]
            ] = defaultdict(list)
            # Determine language order from input paths
            lang_order: List[str] = []
            seen_langs: set[str] = set()
            size_set: set[int] = set()
            for p in group_paths:
                exp = load_exp_info(p)
                lang = lang_key_from_exp(exp)
                if lang not in seen_langs:
                    seen_langs.add(lang)
                    lang_order.append(lang)
                size_set.add(exp["concat_size"])
            size_order: List[int] = sorted(size_set)

            base_paths: List[str | Path] = []
            for lang in lang_order:
                for size in size_order:
                    cfg_key = (size, lang)
                    if cfg_key in config_groups:
                        filtered_config_groups[cfg_key] = config_groups[cfg_key]
                        if config_groups[cfg_key]:
                            base_paths.append(config_groups[cfg_key][0][0])

            subplotter_obj = _CalibrationSubplotter(
                all_results,
                filtered_config_groups,
                label_colors,
                global_ylim,
                global_token_ylim,
            )

            analyze_and_plot_multiple_results(
                paths=base_paths,
                analysis_type=analysis_type,
                document_embedding_type=document_embedding_type,
                title=title,
                pooling_legend_type="segment_standalone_and_document",
                subplotter=subplotter_obj.plot_position_similarities_in_subplot,
                pooling_strategy_segment_standalone=pooling_strategy_segment_standalone,
                pooling_strategy_document=pooling_strategy_document,
                matryoshka_dimensions=None,
                show_segment_lengths=show_segment_lengths,
                show_lengths=show_lengths,
                figure_width=figure_width,
                subplot_height=subplot_height,
                save_plot=False,
                save_path=None,
                show_plot=False,
                return_full_results=False,
                single_model_mode=(
                    True
                    # True if single_model_mode is None else single_model_mode
                ),
                suppress_default_legend=True,
                grid_hspace_override=0.4,  # hspace calibration plot
            )

            # Inject calibration legend at the bottom
            fig = plt.gcf()

            # Hacky adjustment: match multi-model aspect ratio by stretching the figure height
            target_rows_for_ratio = 4
            current_rows = len(size_order)
            if current_rows > 0 and target_rows_for_ratio > 0:
                height_scale = target_rows_for_ratio / current_rows
                fig.set_size_inches(
                    fig.get_figwidth(),
                    fig.get_figheight() * height_scale,
                )
            handles = []
            for lab in unique_labels:
                linestyle = "--" if lab.startswith("Doc_only_clb") else "-"
                h = mlines.Line2D(
                    [],
                    [],
                    color=label_colors[lab],
                    marker="o",
                    # markersize=PLOT_MARKERSIZE,
                    linestyle=linestyle,
                    linewidth=LEGEND_LINEWIDTH,
                    label=lab,
                )
                handles.append(h)
            ci_color = label_colors.get(
                "Original mGTE (no attention calibration)",
                "blue",
            )
            ci_patch = mpatches.Patch(
                color=ci_color, alpha=0.2, label="95% Confidence Interval"
            )
            handles.append(ci_patch)
            assert handles, "Legend handles must not be empty."
            ncol = max(1, (len(handles) + 1) // 2)
            # Align subplot spacing with multi-model plots while reserving space for the legend
            fig.subplots_adjust(
                left=0.02,
                right=0.98,
                top=0.88,
                bottom=0.17,
                wspace=SUBPLOT_COLUMN_SPACING,
                hspace=0.32,
            )
            fig.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.04),
                ncol=ncol,
                fontsize=27,
                frameon=False,
                columnspacing=2.0,
                handletextpad=1.2,
                markerscale=2.5,
            )

            if save_plot:
                out_path = per_save_path
                if out_path is None:
                    filename = f"multi_position_analysis_{pooling_strategy_segment_standalone}_{pooling_strategy_document}.png"
                    out_path = os.path.join(str(group_paths[0]), filename)
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                print(f"Plot saved to {out_path}")
            if show_plot:
                plt.show()

        # Handle optional splitting into multiple figures
        if not split_plots_by_target_lang and not split_plots_by_source_lang:
            render_one_figure(paths, save_path)
            return None

        # Build groups
        groups: Dict[str, List[str | Path]] = defaultdict(list)
        for p in paths:
            exp = load_exp_info(p)
            if split_plots_by_target_lang:
                key = (
                    exp.get("target_lang")
                    if exp.get("target_lang") is not None
                    else "mono"
                )
            else:
                key = exp.get("source_lang", "unknown")
            groups[str(key)].append(p)

        for grp_key, grp_paths in groups.items():
            per_save_path = None
            if save_plot and save_path is not None:
                base, ext = os.path.splitext(str(save_path))
                suffix = (
                    f"_tgt-{grp_key}"
                    if split_plots_by_target_lang
                    else f"_src-{grp_key}"
                )
                per_save_path = f"{base}{suffix}{ext}"
            render_one_figure(grp_paths, per_save_path)

        return None

    def _analyze_and_plot_multiple_results_multi_model(
        self,
        paths: List[str | Path],
        model_pooling_strats: Dict[str, str],
        analysis_type: str = "position",
        document_embedding_type: str = "document-level",
        title: str = "Similarity between Document-Level Embedding and Standalone Segment Embeddings",
        pooling_legend_type: str = "segment_standalone_and_document",
        subplotter: Callable = None,
        pooling_strategy_segment_standalone: str = "cls",
        pooling_strategy_document: str = "cls",
        matryoshka_dimensions: Optional[List[int]] = None,
        merge_latechunk_jina: bool = False,
        show_segment_lengths: bool = False,
        show_lengths: bool = False,
        figure_width: int = 15,
        subplot_height: int = 4,
        save_plot: bool = False,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        return_full_results: bool = False,
        single_model_mode: Optional[bool] = None,
        global_ylim_override: Optional[Tuple[float, float]] = None,
        global_token_ylim_override: Optional[Tuple[float, float]] = None,
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Modified version of analyze_and_plot_multiple_results that shows different models instead of matryoshka dimensions.
        """
        if not subplotter:
            raise ValueError("A subplotter function must be provided.")

        # Setup analysis
        doc_seg_analyzer = DocumentSegmentSimilarityAnalyzer()

        # Get unique experiment configs (excluding model)
        from ..analysis.segment_embedding_analysis import load_exp_info

        config_groups = defaultdict(list)
        for path in paths:
            exp_info = load_exp_info(path)
            # Use source_lang and target_lang from the experiment info for proper language identification
            source_lang = exp_info.get("source_lang", "unknown")
            target_lang = exp_info.get("target_lang", None)

            # Create language key that distinguishes between monolingual and multilingual experiments
            if target_lang is None:
                lang_key = source_lang  # Monolingual: e.g., "de"
            else:
                lang_key = f"{source_lang}_{target_lang}"  # Multilingual: e.g., "de_en"

            config_key = (exp_info["concat_size"], lang_key)
            config_groups[config_key].append((path, exp_info["model_name"]))

        # Pre-compute ALL results for ALL models to get proper y-limits
        all_model_results = {}
        all_y_values = []

        latechunk_overlay_label = "jina-v3"
        latechunk_overlay_color = "lightsalmon"
        latechunk_overlay_results: Dict[Tuple[int, str], Dict[str, Any]] = {}

        for config_key, path_model_pairs in config_groups.items():
            for path, model_name in path_model_pairs:
                pooling_strategy = model_pooling_strats.get(model_name, "cls")

                result = doc_seg_analyzer.run_position_analysis(
                    path=path,
                    document_embedding_type=document_embedding_type,
                    pooling_strategy_segment_standalone=pooling_strategy,
                    pooling_strategy_document=pooling_strategy,
                    matryoshka_dimensions=None,
                )

                # Store result for later use
                result_key = (config_key, model_name)
                all_model_results[result_key] = result

                # Collect y-values for global scaling
                all_y_values.extend(result["position_means"])
                all_y_values.extend(result["position_ci_lower"])
                all_y_values.extend(result["position_ci_upper"])

            if merge_latechunk_jina:
                jina_paths = [
                    (p, m)
                    for (p, m) in path_model_pairs
                    if m == "jinaai/jina-embeddings-v3"
                ]
                assert (
                    len(jina_paths) == 1
                ), f"Expected exactly one Jina path for config_key={config_key}, got {len(jina_paths)}"
                jina_path, jina_model_name = jina_paths[0]
                jina_pooling = model_pooling_strats.get(jina_model_name, "cls")
                latechunk_result = doc_seg_analyzer.run_position_analysis(
                    path=jina_path,
                    document_embedding_type="latechunk-segment",
                    pooling_strategy_segment_standalone=jina_pooling,
                    pooling_strategy_document="cls",
                    matryoshka_dimensions=None,
                )
                latechunk_overlay_results[config_key] = latechunk_result
                all_y_values.extend(latechunk_result["position_means"])
                all_y_values.extend(latechunk_result["position_ci_lower"])
                all_y_values.extend(latechunk_result["position_ci_upper"])

        # Calculate global y-limits with margin (or use override)
        if global_ylim_override is not None:
            global_ylim = global_ylim_override
        else:
            if all_y_values:
                y_min = min(all_y_values)
                y_max = max(all_y_values)
                y_range = y_max - y_min
                margin = y_range * 0.05
                global_ylim = (y_min - margin, y_max + margin)
            else:
                global_ylim = None

        # Calculate global token y-limits (or use override)
        if global_token_ylim_override is not None:
            global_token_ylim = global_token_ylim_override
        elif show_lengths:
            token_len_results = compute_position_token_lengths(paths)
            token_vals: List[float] = []
            for v in token_len_results.values():
                token_vals.extend(v.get("position_means", []))
            if token_vals:
                tmax = max(token_vals)
                tmargin = tmax * 0.05
                global_token_ylim = (0, tmax + tmargin)
            else:
                global_token_ylim = None
        else:
            global_token_ylim = None

        # Create a custom subplotter that uses pre-computed results
        class CustomMultiModelSubplotter:
            def __init__(
                self,
                original_subplotter_obj,
                all_model_results,
                config_groups,
                model_pooling_strats,
                global_ylim,
                global_token_ylim,
                latechunk_overlay_results,
                latechunk_overlay_label,
                latechunk_overlay_color,
            ):
                self.original_subplotter_obj = original_subplotter_obj
                self.all_model_results = all_model_results
                self.config_groups = config_groups
                self.model_pooling_strats = model_pooling_strats
                self.global_ylim = global_ylim
                self.global_token_ylim = global_token_ylim
                self.latechunk_overlay_results = latechunk_overlay_results
                self.latechunk_overlay_label = latechunk_overlay_label
                self.latechunk_overlay_color = latechunk_overlay_color

            def plot_position_similarities_in_subplot(self, ax, base_result, **kwargs):
                # Override ylim with our global one
                kwargs["ylim"] = self.global_ylim
                # Ensure token length axis uses identical limits across figures when provided
                if self.global_token_ylim is not None:
                    kwargs["token_ylim"] = self.global_token_ylim
                return self.original_subplotter_obj.plot_position_similarities_in_subplot_with_precomputed(
                    ax,
                    base_result,
                    self.all_model_results,
                    self.config_groups,
                    self.model_pooling_strats,
                    latechunk_overlay_results=self.latechunk_overlay_results,
                    latechunk_overlay_label=self.latechunk_overlay_label,
                    latechunk_overlay_color=self.latechunk_overlay_color,
                    **kwargs,
                )

        custom_subplotter = CustomMultiModelSubplotter(
            subplotter,
            all_model_results,
            config_groups,
            model_pooling_strats,
            global_ylim,
            global_token_ylim,
            latechunk_overlay_results if merge_latechunk_jina else None,
            latechunk_overlay_label,
            latechunk_overlay_color,
        )

        # Run analysis for each unique config (use first model for each config as base)
        base_results = []
        for config_key, path_model_pairs in config_groups.items():
            # Use first path as base result
            base_path, base_model_name = path_model_pairs[0]
            result_key = (config_key, base_model_name)
            result = all_model_results[result_key]
            base_results.append(result)

        # Use the existing analyze_and_plot_multiple_results logic but with custom legend and ylim
        return analyze_and_plot_multiple_results(
            paths=[
                pair[0] for pairs in config_groups.values() for pair in pairs[:1]
            ],  # Use first path from each config
            analysis_type=analysis_type,
            document_embedding_type=document_embedding_type,
            title=title,
            pooling_legend_type=pooling_legend_type,
            subplotter=custom_subplotter.plot_position_similarities_in_subplot,
            pooling_strategy_segment_standalone=pooling_strategy_segment_standalone,
            pooling_strategy_document=pooling_strategy_document,
            matryoshka_dimensions=None,
            show_segment_lengths=show_segment_lengths,
            show_lengths=show_lengths,
            figure_width=figure_width,
            subplot_height=subplot_height,
            save_plot=save_plot,
            save_path=save_path,
            show_plot=show_plot,
            return_full_results=return_full_results,
            single_model_mode=single_model_mode,
        )


class SegmentLatechunk2SegmentStandaloneSimPlotter:
    """
    Class to handle the plotting of position analysis results in a multi plot.
    """

    def __init__(self):
        self.analysis_type = "position"
        self.document_embedding_type = "latechunk-segment"

    def plot(
        self,
        paths: List[str | Path],
        pooling_strategy_segment_standalone: str = "cls",
        pooling_strategy_document: str = "cls",
        matryoshka_dimensions: Optional[List[int]] = None,
        show_segment_lengths: bool = False,
        show_lengths: bool = False,
        figure_width: int = 15,
        subplot_height: int = 5,
        save_plot: bool = False,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        return_full_results: bool = False,
        single_model_mode: Optional[bool] = None,
        split_plots_by_source_lang: bool = False,
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Analyze and plot position-based similarity metrics for multiple experiments in a grid,
        organized by model name and concat size.

        Args:
            paths: List of paths to experiment results
            pooling_strategy_segment_standalone: Either "cls" or "mean" for standalone segment embeddings pooling
            pooling_strategy_document: Either "cls" or "mean" for document embeddings pooling
            matryoshka_dimensions: Optional list of dimensions to truncate embeddings to for Matryoshka analysis
            show_segment_lengths: Whether to show segment length information in titles and legends (default: True)
            figure_width: Width of the complete figure in inches
            subplot_height: Height of each subplot in inches
            save_plot: Whether to save the plot
            save_path: Path to save the figure (if None and save_plot is True, saves to first path directory)
            show_plot: Whether to display the plot
            return_full_results: Whether to return the full analysis results

        Returns:
            If return_full_results is True, returns a dictionary with model names as keys and
            lists of result dictionaries as values
        """

        position_similarity_single_plotter = PositionSimilaritySinglePlotter()

        # If not splitting, keep existing behavior
        if not split_plots_by_source_lang:
            return analyze_and_plot_multiple_results(
                paths=paths,
                analysis_type=self.analysis_type,
                document_embedding_type=self.document_embedding_type,
                title="Similarity between Contextualized and Standalone Segment Embeddings",
                pooling_legend_type="segment_standalone",
                subplotter=position_similarity_single_plotter.plot_position_similarities_in_subplot,
                pooling_strategy_segment_standalone=pooling_strategy_segment_standalone,
                pooling_strategy_document=pooling_strategy_document,
                matryoshka_dimensions=matryoshka_dimensions,
                show_segment_lengths=show_segment_lengths,
                show_lengths=show_lengths,
                figure_width=figure_width,
                subplot_height=subplot_height,
                save_plot=save_plot,
                save_path=save_path,
                show_plot=show_plot,
                return_full_results=return_full_results,
                single_model_mode=single_model_mode,
            )

        # Split by source_lang with identical formatting across all figures
        from ..analysis.segment_embedding_analysis import load_exp_info

        # Group paths by source_lang
        groups: Dict[str, List[str | Path]] = defaultdict(list)
        for p in paths:
            exp = load_exp_info(p)
            key = exp.get("source_lang", "unknown")
            groups[str(key)].append(p)

        # Pre-compute global y-limits across ALL paths for identical scaling
        doc_seg_analyzer = DocumentSegmentSimilarityAnalyzer()
        all_y_values: List[float] = []
        for p in paths:
            res = doc_seg_analyzer.run_position_analysis(
                path=p,
                document_embedding_type=self.document_embedding_type,
                pooling_strategy_segment_standalone=pooling_strategy_segment_standalone,
                pooling_strategy_document=pooling_strategy_document,
                matryoshka_dimensions=None,
            )
            all_y_values.extend(res["position_means"])
            all_y_values.extend(res["position_ci_lower"])
            all_y_values.extend(res["position_ci_upper"])

        global_ylim: Optional[Tuple[float, float]] = None
        if all_y_values:
            y_min = min(all_y_values)
            y_max = max(all_y_values)
            y_range = y_max - y_min
            margin = y_range * 0.05
            global_ylim = (y_min - margin, y_max + margin)

        # Pre-compute global token length y-limits if requested
        global_token_ylim: Optional[Tuple[float, float]] = None
        if show_lengths:
            token_len_results = compute_position_token_lengths(paths)
            token_vals: List[float] = []
            for v in token_len_results.values():
                token_vals.extend(v.get("position_means", []))
            if token_vals:
                tmax = max(token_vals)
                tmargin = tmax * 0.05
                global_token_ylim = (0, tmax + tmargin)

        # Wrapper subplotter to enforce global limits
        class _FixedLimitsSubplotter:
            def __init__(self, original, ylim, token_ylim):
                self._orig = original
                self._ylim = ylim
                self._token_ylim = token_ylim

            def plot_position_similarities_in_subplot(self, ax, base_result, **kwargs):
                if self._ylim is not None:
                    kwargs["ylim"] = self._ylim
                if self._token_ylim is not None:
                    kwargs["token_ylim"] = self._token_ylim
                return self._orig(ax, base_result, **kwargs)

        wrapper = _FixedLimitsSubplotter(
            position_similarity_single_plotter.plot_position_similarities_in_subplot,
            global_ylim,
            global_token_ylim,
        )

        merged_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for src_key, src_paths in groups.items():
            per_save_path = None
            if save_plot and save_path is not None:
                base, ext = os.path.splitext(str(save_path))
                per_save_path = f"{base}_src-{src_key}{ext}"

            out = analyze_and_plot_multiple_results(
                paths=src_paths,
                analysis_type=self.analysis_type,
                document_embedding_type=self.document_embedding_type,
                title="Similarity between Contextualized and Standalone Segment Embeddings",
                pooling_legend_type="segment_standalone",
                subplotter=wrapper.plot_position_similarities_in_subplot,
                pooling_strategy_segment_standalone=pooling_strategy_segment_standalone,
                pooling_strategy_document=pooling_strategy_document,
                matryoshka_dimensions=matryoshka_dimensions,
                show_segment_lengths=show_segment_lengths,
                show_lengths=show_lengths,
                figure_width=figure_width,
                subplot_height=subplot_height,
                save_plot=save_plot,
                save_path=per_save_path,
                show_plot=show_plot,
                return_full_results=return_full_results,
                single_model_mode=single_model_mode,
            )

            if return_full_results and isinstance(out, dict) and "model_results" in out:
                for k, v in out["model_results"].items():
                    merged_results[k].extend(v)

        return {"model_results": dict(merged_results)} if return_full_results else None


class PositionalDirectionalLeakageMultiPlotter:
    """
    Class to handle the plotting of positional directional leakage analysis results in a multi plot.

    NOTE: The averages computed by this plotter may differ slightly from DirectionalLeakageMultiPlotter
    due to different averaging methodologies:
    - This plotter: Averages per-position means (equal weight per position)
    - DirectionalLeakageMultiPlotter: Simple mean over all pairwise similarities

    """

    def __init__(self):
        self.analysis_type = "positional_directional_leakage"

    def plot(
        self,
        paths: List[str | Path],
        pooling_strategy_segment_standalone: str = "mean",
        matryoshka_dimensions: Optional[List[int]] = None,
        show_segment_lengths: bool = False,
        figure_width: int = 15,
        subplot_height: int = 5,
        save_plot: bool = False,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        return_full_results: bool = False,
        single_model_mode: Optional[bool] = None,
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Analyze and plot positional directional leakage for multiple experiments in a grid,
        organized by model name and concat size.

        Args:
            paths: List of paths to experiment results
            pooling_strategy_segment_standalone: Either "cls" or "mean" for standalone segment embeddings pooling
            matryoshka_dimensions: Optional list of dimensions to truncate embeddings to for Matryoshka analysis
            show_segment_lengths: Whether to show segment length information in titles and legends
            figure_width: Width of the complete figure in inches
            subplot_height: Height of each subplot in inches
            save_plot: Whether to save the plot
            save_path: Path to save the figure (if None and save_plot is True, saves to first path directory)
            show_plot: Whether to display the plot
            return_full_results: Whether to return the full analysis results
            single_model_mode: If True, optimize layout for single model. If None, auto-detect based on data

        Returns:
            If return_full_results is True, returns a dictionary with model names as keys and
            lists of result dictionaries as values
        """
        positional_directional_leakage_single_plotter = (
            PositionalDirectionalLeakageSinglePlotter()
        )

        return analyze_and_plot_multiple_results(
            paths=paths,
            analysis_type=self.analysis_type,
            title="Position-wise Influence on Segment Contextualization",
            pooling_legend_type="segment_standalone",
            subplotter=positional_directional_leakage_single_plotter.plot_positional_directional_leakage_in_subplot,
            pooling_strategy_segment_standalone=pooling_strategy_segment_standalone,
            matryoshka_dimensions=matryoshka_dimensions,
            show_segment_lengths=show_segment_lengths,
            figure_width=figure_width,
            subplot_height=subplot_height,
            save_plot=save_plot,
            save_path=save_path,
            show_plot=show_plot,
            return_full_results=return_full_results,
            single_model_mode=single_model_mode,
        )


class PositionalDirectionalLeakageHeatmapMultiPlotter:
    """
    Class to handle the plotting of positional directional leakage analysis results as heatmaps.
    Each subplot shows a matrix where:
    - Upper triangle: forward similarities (position i -> position j, where j > i)
    - Lower triangle: backward similarities (position i -> position j, where i > j)
    - Diagonal: not applicable (set to NaN)

    This provides detailed position-wise similarity information rather than averaged values.
    """

    def __init__(self):
        self.analysis_type = "positional_directional_leakage_heatmap"

    def plot(
        self,
        paths: List[str | Path],
        pooling_strategy_segment_standalone: str = "mean",
        matryoshka_dimensions: Optional[List[int]] = None,
        show_segment_lengths: bool = False,
        figure_width: int = 15,
        subplot_height: int = 5,
        save_plot: bool = False,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        return_full_results: bool = False,
        single_model_mode: Optional[bool] = None,
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Analyze and plot positional directional leakage heatmaps for multiple experiments in a grid,
        organized by model name and concat size.

        Args:
            paths: List of paths to experiment results
            pooling_strategy_segment_standalone: Either "cls" or "mean" for standalone segment embeddings pooling
            matryoshka_dimensions: Optional list of dimensions to truncate embeddings to for Matryoshka analysis
            show_segment_lengths: Whether to show segment length information in titles and legends
            figure_width: Width of the complete figure in inches
            subplot_height: Height of each subplot in inches
            save_plot: Whether to save the plot
            save_path: Path to save the figure (if None and save_plot is True, saves to first path directory)
            show_plot: Whether to display the plot
            return_full_results: Whether to return the full analysis results
            single_model_mode: If True, optimize layout for single model. If None, auto-detect based on data

        Returns:
            If return_full_results is True, returns a dictionary with model names as keys and
            lists of result dictionaries as values
        """
        heatmap_single_plotter = PositionalDirectionalLeakageHeatmapSinglePlotter()

        return analyze_and_plot_multiple_results(
            paths=paths,
            analysis_type=self.analysis_type,
            title="Position-wise Directional Leakage Heatmaps",
            pooling_legend_type="segment_standalone",
            subplotter=heatmap_single_plotter.plot_heatmap_in_subplot,
            pooling_strategy_segment_standalone=pooling_strategy_segment_standalone,
            matryoshka_dimensions=matryoshka_dimensions,
            show_segment_lengths=show_segment_lengths,
            figure_width=figure_width,
            subplot_height=subplot_height,
            save_plot=save_plot,
            save_path=save_path,
            show_plot=show_plot,
            return_full_results=return_full_results,
            single_model_mode=single_model_mode,
        )


def analyze_and_plot_multiple_results(
    paths: List[str | Path],
    analysis_type: str = "position",  # Either "position" or "directional_leakage"
    document_embedding_type: str = "document-level",
    title: str = "Similarity between Document-Level Embedding and Standalone Segment Embeddings",
    pooling_legend_type: str = "segment_standalone_and_document",  # Either "segment_standalone" or "segment_standalone_and_document"
    subplotter: Callable = None,
    pooling_strategy_segment_standalone: str = "cls",
    pooling_strategy_document: str = "cls",
    matryoshka_dimensions: Optional[List[int]] = None,
    show_segment_lengths: bool = False,
    show_lengths: bool = False,
    figure_width: int = 15,
    subplot_height: int = 4,
    save_plot: bool = False,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    return_full_results: bool = False,
    single_model_mode: Optional[bool] = None,
    suppress_default_legend: bool = False,
    grid_hspace_override: Optional[float] = None,
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """
    Analyze and plot multiple experiment results in a grid, organized by model name and concat size.
    Supports position similarity, directional leakage, and heatmap analysis.

    Args:
        paths: List of paths to experiment results
        analysis_type: Type of analysis to perform ("position", "directional_leakage",
                      "positional_directional_leakage", or "positional_directional_leakage_heatmap")
        document_embedding_type: Type of document embedding to use
        title: Title for the plot
        pooling_legend_type: Type of pooling legend to show
        subplotter: Function to use for plotting individual subplots
        pooling_strategy_segment_standalone: Either "cls" or "mean" for standalone segment embeddings pooling
        pooling_strategy_document: Either "cls" or "mean" for document embeddings pooling
                                      (only used for position analysis)
        matryoshka_dimensions: Optional list of dimensions to truncate embeddings to for Matryoshka analysis
        show_segment_lengths: Whether to show segment length information in titles and legends (default: True)
        show_lengths: Whether to show token lengths as bar charts on right y-axis (default: False)
        figure_width: Width of the complete figure in inches
        subplot_height: Height of each subplot in inches
        save_plot: Whether to save the plot
        save_path: Path to save the figure (if None and save_plot is True, saves to first path directory)
        show_plot: Whether to display the plot
        return_full_results: Whether to return the full analysis results
        single_model_mode: If True, optimize layout for single model. If None, auto-detect based on data
        grid_hspace_override: Optional value to override the GridSpec hspace spacing between subplot rows

    Returns:
        If return_full_results is True, returns a dictionary with model names as keys and
        lists of result dictionaries as values
    """
    if not subplotter:
        raise ValueError("A subplotter function must be provided.")
    # Setup analysis functions based on type
    if analysis_type == "position":
        doc_seg_analyzer = DocumentSegmentSimilarityAnalyzer()
        run_analysis = lambda path: doc_seg_analyzer.run_position_analysis(
            path=path,
            document_embedding_type=document_embedding_type,
            pooling_strategy_segment_standalone=pooling_strategy_segment_standalone,
            pooling_strategy_document=pooling_strategy_document,
            matryoshka_dimensions=matryoshka_dimensions,
        )
        plot_in_subplot = subplotter
    elif analysis_type == "directional_leakage":
        directional_leakage_analyzer = DirectionalLeakageAnalyzer()
        run_analysis = (
            lambda path: directional_leakage_analyzer.run_directional_leakage_analysis(
                path, pooling_strategy_segment_standalone
            )
        )
        plot_in_subplot = subplotter
    elif analysis_type == "positional_directional_leakage":
        positional_directional_leakage_analyzer = PositionalDirectionalLeakageAnalyzer()
        run_analysis = lambda path: positional_directional_leakage_analyzer.run_positional_directional_leakage_analysis(
            path, pooling_strategy_segment_standalone, matryoshka_dimensions
        )
        plot_in_subplot = subplotter
    elif analysis_type == "positional_directional_leakage_heatmap":
        positional_directional_leakage_analyzer = PositionalDirectionalLeakageAnalyzer()
        run_analysis = lambda path: positional_directional_leakage_analyzer.run_positional_directional_leakage_analysis(
            path, pooling_strategy_segment_standalone, matryoshka_dimensions
        )
        plot_in_subplot = subplotter
    else:
        raise ValueError(
            f"Unknown analysis_type: {analysis_type}. Use 'position', 'directional_leakage', 'positional_directional_leakage', or 'positional_directional_leakage_heatmap'."
        )

    # Run analysis for each path
    all_results = []
    for path in paths:
        # print(f"Processing {path}...")
        results = run_analysis(path)
        all_results.append(results)

    # Compute token lengths if requested
    token_length_results = {}
    if show_lengths:
        token_length_results = compute_position_token_lengths(paths)

    # Group results by model name
    model_groups = defaultdict(list)
    for result in all_results:
        model_name = result["model_name"]
        # Abbreviate model names
        if "Alibaba-NLP/gte-multilingual-base" in model_name:
            model_name = "mGTE"
        elif "jinaai/jina-embeddings-v3" in model_name:
            model_name = "jina-v3"
        elif "Qwen/Qwen3-Embedding-0.6B" in model_name:
            model_name = "qwen3-0.6B"
        # Store result with abbreviated model name
        result["abbreviated_model_name"] = model_name

        # Add token length data if available
        if show_lengths and str(result["path"]) in token_length_results:
            result["token_lengths"] = token_length_results[str(result["path"])]

        model_groups[model_name].append(result)

    # Sort results within each group by concat size
    for model_name, results in model_groups.items():
        results.sort(key=lambda x: x["concat_size"])

    # Dictionary to store ranges information for the legend
    ranges_info = {}
    range_id = 1

    # Assign range IDs to each unique range configuration
    for results in model_groups.values():
        for result in results:
            ranges_tuple = tuple(map(tuple, result["position_specific_ranges"]))
            if ranges_tuple and ranges_tuple not in ranges_info:
                ranges_info[ranges_tuple] = f"SL{range_id}"
                result["range_id"] = f"SL{range_id}"
                range_id += 1
            elif ranges_tuple:
                result["range_id"] = ranges_info[ranges_tuple]
            else:
                result["range_id"] = "N/A"

    # Get all unique concat sizes across all models
    all_concat_sizes = set()
    for results in model_groups.values():
        for result in results:
            all_concat_sizes.add(result["concat_size"])
    all_concat_sizes = sorted(all_concat_sizes)
    concat_size_to_row = {size: idx for idx, size in enumerate(all_concat_sizes)}
    num_rows = len(all_concat_sizes)

    # Determine max number of range variations per model per concat size
    max_variations_per_model_size = {}
    for model_name, results in model_groups.items():
        concat_size_groups = defaultdict(list)
        for result in results:
            concat_size_groups[result["concat_size"]].append(result)
        for concat_size, size_results in concat_size_groups.items():
            key = (model_name, concat_size)
            max_variations_per_model_size[key] = len(size_results)

    # Calculate global y-limits for each model (for position and positional directional leakage analysis)
    model_ylims = {}
    if analysis_type == "position":
        for model_name, results in model_groups.items():
            all_values = []
            for result in results:
                # Collect all y-values (means, CI lower, CI upper)
                all_values.extend(result["position_means"])
                all_values.extend(result["position_ci_lower"])
                all_values.extend(result["position_ci_upper"])

                # Also collect Matryoshka results if available
                if "matryoshka_results" in result:
                    for dim_results in result["matryoshka_results"].values():
                        all_values.extend(dim_results["position_means"])
                        all_values.extend(dim_results["position_ci_lower"])
                        all_values.extend(dim_results["position_ci_upper"])

            if all_values:
                # Add a small margin (5%) to the range
                y_min = min(all_values)
                y_max = max(all_values)
                y_range = y_max - y_min
                margin = y_range * 0.05
                model_ylims[model_name] = (y_min - margin, y_max + margin)
            else:
                model_ylims[model_name] = None
    elif analysis_type == "positional_directional_leakage":
        for model_name, results in model_groups.items():
            all_values = []
            for result in results:
                # Collect all y-values from position forward and backward means
                all_values.extend(result["position_forward_means"].values())
                all_values.extend(result["position_backward_means"].values())

                # Also collect Matryoshka results if available
                if "matryoshka_results" in result:
                    for dim_results in result["matryoshka_results"].values():
                        all_values.extend(
                            dim_results["position_forward_means"].values()
                        )
                        all_values.extend(
                            dim_results["position_backward_means"].values()
                        )

            if all_values:
                # Add a small margin (5%) to the range and respect cosine similarity bounds [-1, 1]
                y_min = max(min(all_values) - 0.02, -1.0)
                y_max = min(max(all_values) + 0.02, 1.0)
                model_ylims[model_name] = (y_min, y_max)
            else:
                model_ylims[model_name] = None
    elif analysis_type == "positional_directional_leakage_heatmap":
        # For heatmaps, we don't need ylims since we use imshow
        # The colormap will handle the scaling automatically with vmin/vmax
        for model_name, results in model_groups.items():
            model_ylims[model_name] = None

    # Calculate global token length limits for each model (for show_lengths)
    model_token_ylims = {}
    if show_lengths:
        for model_name, results in model_groups.items():
            all_token_lengths = []
            for result in results:
                if "token_lengths" in result:
                    token_data = result["token_lengths"]
                    all_token_lengths.extend(token_data["position_means"])

            if all_token_lengths:
                # Add a small margin (5%) to the range
                y_min = min(all_token_lengths)
                y_max = max(all_token_lengths)
                y_range = y_max - y_min
                margin = y_range * 0.05
                model_token_ylims[model_name] = (max(0, y_min - margin), y_max + margin)
            else:
                model_token_ylims[model_name] = None

    # Calculate global x-limits for each model (for directional leakage analysis only)
    model_xlims = {}
    if analysis_type == "directional_leakage":
        for model_name, results in model_groups.items():
            all_values = []
            for result in results:
                # Collect all x-values (forward and backward influence values)
                all_values.extend(result["forward_influence"])
                all_values.extend(result["backward_influence"])

            if all_values:
                # Add a small margin and respect cosine similarity bounds [-1, 1]
                x_min = max(min(all_values) - 0.01, -1.0)
                x_max = min(max(all_values) + 0.01, 1.0)
                model_xlims[model_name] = (x_min, x_max)
            else:
                model_xlims[model_name] = None

    # Calculate total columns needed (sum of max variations per model + spacing)
    total_cols = 0
    model_col_start_indices = {}
    model_col_counts = {}  # Store the width of each model's columns
    spacing_offset = 0
    for model_name in model_groups.keys():
        model_col_start_indices[model_name] = total_cols + spacing_offset
        max_cols_for_model = max(
            max_variations_per_model_size.get((model_name, size), 0)
            for size in all_concat_sizes
        )
        model_col_counts[model_name] = max_cols_for_model
        total_cols += max_cols_for_model
        # Add spacing column after each model except the last one
        if model_name != list(model_groups.keys())[-1]:
            spacing_offset += 1

    # Detect if we have only one model for layout optimization
    if single_model_mode is None:
        # Auto-detect based on data
        single_model_mode = len(model_groups.keys()) == 1
    # If explicitly set by user, use that value

    # Create figure with GridSpec for flexible subplot layout
    if single_model_mode:
        # For single model, make subplots square and use full width
        # Calculate optimal dimensions based on number of columns
        max_cols_for_single_model = max(
            max_variations_per_model_size.get((list(model_groups.keys())[0], size), 0)
            for size in all_concat_sizes
        )
        # Make figure wider and shorter to accommodate square subplots
        fig = plt.figure(
            figsize=(
                max_cols_for_single_model
                * 4.5,  # Much wider - about 4.5 inches per column
                num_rows * 3.5 * SUBPLOT_VERTICAL_SCALE,
            )
        )
    else:
        fig = plt.figure(
            figsize=(
                figure_width * len(model_groups.keys()) * 1.0,
                subplot_height * num_rows * SUBPLOT_VERTICAL_SCALE,
            )
        )

    # Add spacing between models using width_ratios
    width_ratios = []
    for model_name in model_groups.keys():
        max_cols_for_model = max(
            max_variations_per_model_size.get((model_name, size), 0)
            for size in all_concat_sizes
        )
        width_ratios.extend([1] * max_cols_for_model)
        # Add an extra column with moderate width for spacing after each model except the last one
        if model_name != list(model_groups.keys())[-1]:
            width_ratios.append(0.15)  # Spacing between models

    # Adjust total_cols to account for spacing columns
    spacing_cols = len(model_groups) - 1
    total_cols_with_spacing = total_cols + spacing_cols

    # Create GridSpec with custom width ratios and spacing optimized for single/multi model
    if single_model_mode:
        # For single model, optimize spacing for square subplots
        # Can use less spacing since we're hiding redundant axis labels
        gs = GridSpec(
            num_rows,
            total_cols_with_spacing,
            figure=fig,
            width_ratios=width_ratios,
            wspace=SUBPLOT_COLUMN_SPACING,
            hspace=grid_hspace_override if grid_hspace_override is not None else 0.45,
        )
    else:
        # For multiple models, use optimized spacing since axis labels are now hidden
        gs = GridSpec(
            num_rows,
            total_cols_with_spacing,
            figure=fig,
            width_ratios=width_ratios,
            wspace=SUBPLOT_COLUMN_SPACING,
            hspace=grid_hspace_override if grid_hspace_override is not None else 0.45,
        )

    # --- Center model names above their columns ---
    # Calculate the center x-position for each model's columns using axes positions
    # Only show model names if we have multiple models
    if not single_model_mode:
        for model_idx, (model_name, results) in enumerate(model_groups.items()):
            col_start = model_col_start_indices[model_name]
            model_width = model_col_counts[model_name]
            # Find the leftmost and rightmost axes for this model in the top row
            top_row = 0
            left_ax = fig.add_subplot(gs[top_row, col_start])
            right_ax = fig.add_subplot(gs[top_row, col_start + model_width - 1])
            left_bbox = left_ax.get_position()
            right_bbox = right_ax.get_position()
            left_ax.remove()
            right_ax.remove()
            # Center x is the midpoint between left and right axes
            center_x = (left_bbox.x0 + right_bbox.x1) / 2
            fig.text(
                center_x,
                0.93,  # Higher up for more space below model names
                model_name,
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
            )

    # Add concat_size label only once per row (on the left)
    concat_size_label_drawn = set()

    # Calculate the rightmost column that contains actual data (not spacing)
    rightmost_col = -1
    leftmost_col: Optional[int] = None
    for model_name, results in model_groups.items():
        concat_size_groups = defaultdict(list)
        for result in results:
            concat_size_groups[result["concat_size"]].append(result)

        col_start = model_col_start_indices[model_name]
        for concat_size, size_results in concat_size_groups.items():
            for local_col_idx, result in enumerate(size_results):
                col_idx = col_start + local_col_idx
                rightmost_col = max(rightmost_col, col_idx)
                if leftmost_col is None or col_idx < leftmost_col:
                    leftmost_col = col_idx

    for model_idx, (model_name, results) in enumerate(model_groups.items()):
        concat_size_groups = defaultdict(list)
        for result in results:
            concat_size_groups[result["concat_size"]].append(result)

        col_start = model_col_start_indices[model_name]
        model_width = model_col_counts[model_name]

        for concat_size, size_results in concat_size_groups.items():
            row_idx = concat_size_to_row[concat_size]

            for local_col_idx, result in enumerate(size_results):
                col_idx = col_start + local_col_idx

                if local_col_idx >= max_variations_per_model_size.get(
                    (model_name, concat_size), 0
                ):
                    print(
                        f"Warning: Too many results for {model_name} concat size {concat_size}. Skipping extra plots."
                    )
                    break

                ax = fig.add_subplot(gs[row_idx, col_idx])

                # Pass ylim for position analysis or xlim for directional leakage analysis
                if analysis_type == "position":
                    ylim = model_ylims.get(model_name)
                    token_ylim = (
                        model_token_ylims.get(model_name) if show_lengths else None
                    )
                    # Only show token y-axis label on the rightmost subplot
                    show_token_ylabel = (
                        (col_idx == rightmost_col) if show_lengths else True
                    )
                    show_token_ticklabels = (
                        (col_idx == rightmost_col) if show_lengths else True
                    )
                    show_cosine_ticklabels = (
                        True if leftmost_col is None else col_idx == leftmost_col
                    )
                    plot_in_subplot(
                        ax,
                        result,
                        show_title=True,
                        compact=True,
                        ylim=ylim,
                        show_segment_lengths=show_segment_lengths,
                        show_lengths=show_lengths,
                        token_ylim=token_ylim,
                        show_token_ylabel=show_token_ylabel,
                        show_token_ticklabels=show_token_ticklabels,
                        show_cosine_ticklabels=show_cosine_ticklabels,
                    )
                    ax.tick_params(axis="y", labelleft=show_cosine_ticklabels)
                elif analysis_type == "directional_leakage":
                    xlim = model_xlims.get(model_name)
                    plot_in_subplot(
                        ax, result, show_title=True, compact=True, xlim=xlim
                    )
                elif analysis_type == "positional_directional_leakage":
                    ylim = model_ylims.get(model_name)
                    plot_in_subplot(
                        ax,
                        result,
                        show_title=True,
                        compact=True,
                        ylim=ylim,
                        show_segment_lengths=show_segment_lengths,
                    )
                elif analysis_type == "positional_directional_leakage_heatmap":
                    plot_in_subplot(
                        ax,
                        result,
                        show_title=True,
                        compact=True,
                        show_segment_lengths=show_segment_lengths,
                    )
                else:
                    plot_in_subplot(
                        ax,
                        result,
                        show_title=True,
                        compact=True,
                        show_segment_lengths=show_segment_lengths,
                    )

                # Control axis labels visibility
                # Y-axis label: only show on leftmost subplot of each row
                if not (model_idx == 0 and local_col_idx == 0):
                    ax.set_ylabel("")

                # X-axis label: only show on bottom subplot of each column
                if row_idx != num_rows - 1:  # Not the bottom row
                    ax.set_xlabel("")

                # # Only draw concat_size label once per row, on the leftmost subplot
                # if (
                #     row_idx not in concat_size_label_drawn
                #     and model_idx == 0
                #     and local_col_idx == 0
                # ):
                #     # Adjust label position based on single model mode
                #     x_pos = -0.25 if single_model_mode else -0.50
                #     ax.text(
                #         x_pos,  # Closer to subplot for single model mode
                #         0.5,
                #         f"# of Segments: {concat_size}",
                #         verticalalignment="center",
                #         horizontalalignment="right",
                #         transform=ax.transAxes,
                #         fontsize=12,
                #         fontweight="bold",
                #         rotation=90,
                #     )
                #     concat_size_label_drawn.add(row_idx)

            # Fill in any empty spots in the grid with blank subplots
            max_cols_for_this_model_size = max_variations_per_model_size.get(
                (model_name, concat_size), 0
            )
            for local_col_idx in range(len(size_results), max_cols_for_this_model_size):
                col_idx = col_start + local_col_idx
                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.axis("off")

            # Add spacing column after each model except the last one
            if model_name != list(model_groups.keys())[-1]:
                spacer_col_idx = col_start + max_cols_for_this_model_size
                ax = fig.add_subplot(gs[row_idx, spacer_col_idx])
                ax.axis("off")

    # Adjust layout - leave more space at top for main title and bottom for legends
    if single_model_mode:
        # For single model, adjust for wider/shorter layout
        plt.tight_layout(rect=[0.02, 0.08, 0.98, 0.93])  # More margins on all sides
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.85])  # More space at top and bottom

    # Add overall title
    if single_model_mode:
        # For single model, position title higher and include model name in title
        single_model_name = list(model_groups.keys())[0]
        y_title = 0.96  # Adjusted for new layout
        y_subtitle = y_title - 0.035

        if analysis_type == "position":
            # Check if this is a multi-model plot (even if single_model_mode is True due to auto-detection)
            is_multi_model = hasattr(subplotter, "__self__") and hasattr(
                subplotter.__self__, "original_subplotter_obj"
            )

            if is_multi_model:
                # For multi-model plots, don't include model name in title
                fig.suptitle(
                    title,
                    fontsize=31,  # before: 28
                    fontweight="bold",
                    y=y_title,
                )
            else:
                # Include model name in the title for single model
                # title_with_model = f"{title} - {single_model_name}"
                title_with_model = title
                fig.suptitle(
                    title_with_model,
                    fontsize=31,  # before: 28
                    fontweight="bold",
                    y=y_title,
                )
        elif analysis_type in ("directional_leakage", "positional_directional_leakage"):
            # Include model name in the title for single model
            title_with_model = f"{title} - {single_model_name}"
            fig.suptitle(
                title_with_model,
                fontsize=28,
                fontweight="bold",
                y=y_title,
            )

            # Add legend explaining directional leakage for single model
            subtitle_text = (
                "Forward Similarity (F): Earlier segments' standalone embeddings ↔ Later segments' contextualized embeddings\n"
                "Backward Similarity (B): Later segments' standalone embeddings ↔ Earlier segments' contextualized embeddings"
            )
            fig.text(
                0.5,
                y_subtitle,  # Position between title and plots
                subtitle_text,
                ha="center",
                va="top",
                fontsize=16,  # Slightly smaller for single model
            )
    else:
        # For multiple models, use original positioning
        y_title = 1.03
        y_subtitle = y_title - 0.03

        if analysis_type == "position":
            # Split into two lines for different font sizes for position analysis
            fig.suptitle(
                title,
                fontsize=24,
                fontweight="bold",
                y=y_title,
            )
        elif analysis_type in (
            "directional_leakage",
            "positional_directional_leakage",
            "positional_directional_leakage_heatmap",
        ):
            # Directional leakage title
            fig.suptitle(
                title,
                fontsize=24,
                fontweight="bold",
                y=y_title,
            )

        # Add legend explaining directional leakage
        if analysis_type in ("directional_leakage", "positional_directional_leakage"):
            subtitle_text = (
                "Forward Similarity (F): Earlier segments' standalone embeddings ↔ Later segments' contextualized embeddings\n"
                "Backward Similarity (B): Later segments' standalone embeddings ↔ Earlier segments' contextualized embeddings"
            )
            fig.text(
                0.5,
                y_subtitle,  # Position between title and plots
                subtitle_text,
                ha="center",
                va="top",
                fontsize=18,
            )
        elif analysis_type == "positional_directional_leakage_heatmap":
            subtitle_text = (
                "Similarity matrices: Upper triangle = Forward (earlier→later), Lower triangle = Backward (later→earlier)\n"
                "Values represent cosine similarities between standalone and contextualized embeddings"
            )
            fig.text(
                0.5,
                y_subtitle,  # Position between title and plots
                subtitle_text,
                ha="center",
                va="top",
                fontsize=18,
            )

    # Add custom legends based on analysis type
    if not suppress_default_legend:
        if single_model_mode:
            legend_y = 0.01  # Lower position for single model due to new layout
        else:
            legend_y = 0.05

        if analysis_type == "position":
            # Position analysis legend - check if we're using multi-model plotter
            is_multi_model = hasattr(subplotter, "__self__") and hasattr(
                subplotter.__self__, "original_subplotter_obj"
            )

            handles = []
            labels = []

            if is_multi_model:
                # Show model names instead of matryoshka dimensions
                model_colors = {
                    "mGTE": "blue",
                    "jina-v3": "red",
                    # "qwen3-0.6B": "green"
                }

                for model_name, color in model_colors.items():
                    model_line = mlines.Line2D(
                        [],
                        [],
                        color=color,
                        marker="o",
                        linestyle="-",
                        linewidth=LEGEND_LINEWIDTH,
                        label=model_name,
                    )
                    handles.append(model_line)

                overlay_results = getattr(
                    subplotter.__self__, "latechunk_overlay_results", None
                )
                overlay_label = getattr(
                    subplotter.__self__, "latechunk_overlay_label", None
                )
                overlay_color = getattr(
                    subplotter.__self__, "latechunk_overlay_color", None
                )
                if overlay_results is not None:
                    assert (
                        overlay_label is not None and overlay_color is not None
                    ), "Latechunk overlay legend requires label and color."
                    handles.append(
                        mlines.Line2D(
                            [],
                            [],
                            color=overlay_color,
                            marker="o",
                            linestyle="-",
                            linewidth=LEGEND_LINEWIDTH,
                            label=overlay_label,
                        )
                    )
            else:
                # Always include full embedding
                mean_line = mlines.Line2D(
                    [],
                    [],
                    color="red",
                    marker="o",
                    linestyle="-",
                    linewidth=LEGEND_LINEWIDTH,
                    # label="Full Embedding",
                    label="jina-v3",
                )
                handles.append(mean_line)

                # Add Matryoshka dimensions if present (check first result for Matryoshka data)
                has_matryoshka = any(
                    "matryoshka_results" in result for result in all_results
                )
                if has_matryoshka and matryoshka_dimensions:
                    colors = [
                        "red",
                        "green",
                        "orange",
                        "purple",
                        "brown",
                        "pink",
                        "gray",
                        "olive",
                    ]
                    for i, dim in enumerate(matryoshka_dimensions):
                        color = colors[i % len(colors)]
                        mat_line = mlines.Line2D(
                            [],
                            [],
                            color=color,
                            marker="o",
                            linestyle="--" if i >= 4 else "-",
                            linewidth=LEGEND_LINEWIDTH,
                            label=f"Matryoshka D{dim}",
                        )
                        handles.append(mat_line)

            # Add confidence interval patch
            ci_patch = mpatches.Patch(
                color="blue", alpha=0.2, label="95% Confidence Interval"
            )
            handles.append(ci_patch)

            # Determine number of columns based on number of handles
            ncol = min(len(handles), 6)  # Maximum 6 columns to avoid overcrowding

            legend_fontsize = 27  # if is_multi_model else 9
            fig.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, legend_y),
                ncol=ncol,
                fontsize=legend_fontsize,
                frameon=False,
                columnspacing=2.0,
                handletextpad=1.2,
                markerscale=2.5,
            )
        elif analysis_type == "positional_directional_leakage":
            # Position analysis legend - check if Matryoshka dimensions are present
            handles = []
            labels = []

            forward_line = mlines.Line2D(
                [],
                [],
                color="blue",
                marker="o",
                linestyle="-",
                linewidth=2,
                label="Forward Similarity",
            )
            backward_line = mlines.Line2D(
                [],
                [],
                color="orange",
                marker="o",
                linestyle="-",
                linewidth=2,
                label="Backward Similarity",
            )
            handles.append(forward_line)
            handles.append(backward_line)

            # Add Matryoshka dimensions if present (check first result for Matryoshka data)
            has_matryoshka = any(
                "matryoshka_results" in result for result in all_results
            )
            if has_matryoshka and matryoshka_dimensions:
                colors = [
                    "red",
                    "green",
                    "orange",
                    "purple",
                    "brown",
                    "pink",
                    "gray",
                    "olive",
                ]
                for i, dim in enumerate(matryoshka_dimensions):
                    color = colors[i % len(colors)]
                    mat_line = mlines.Line2D(
                        [],
                        [],
                        color=color,
                        marker="o",
                        linestyle="--" if i >= 4 else "-",
                        linewidth=2,
                        label=f"Matryoshka D{dim}",
                    )
                    handles.append(mat_line)

            # Determine number of columns based on number of handles
            ncol = min(len(handles), 6)  # Maximum 6 columns to avoid overcrowding

            fig.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, legend_y),
                ncol=ncol,
                fontsize=9,
                frameon=False,
                columnspacing=2.0,
                handletextpad=1.2,
            )
        elif analysis_type == "positional_directional_leakage_heatmap":
            # Heatmap legend with colorbar explanation
            handles = []

            # Add colorbar explanation patch
            colorbar_patch = mpatches.Patch(
                color="lightblue",
                label="Colorbar: Red = High similarity, Blue = Low similarity",
            )
            handles.append(colorbar_patch)

            # Add matrix explanation patches
            upper_patch = mpatches.Patch(
                color="lightcoral",
                alpha=0.7,
                label="Upper triangle: Forward (earlier→later)",
            )
            lower_patch = mpatches.Patch(
                color="lightgreen",
                alpha=0.7,
                label="Lower triangle: Backward (later→earlier)",
            )
            handles.append(upper_patch)
            handles.append(lower_patch)

            # Add Matryoshka dimensions if present
            has_matryoshka = any(
                "matryoshka_results" in result for result in all_results
            )
            if has_matryoshka and matryoshka_dimensions:
                mat_patch = mpatches.Patch(
                    color="gold",
                    alpha=0.7,
                    label=f"Matryoshka dimensions: {matryoshka_dimensions}",
                )
                handles.append(mat_patch)

            # Determine number of columns based on number of handles
            ncol = min(len(handles), 3)  # Maximum 3 columns for heatmap legends

            fig.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, legend_y),
                ncol=ncol,
                fontsize=9,
                frameon=False,
                columnspacing=2.0,
                handletextpad=1.2,
            )
        elif analysis_type == "directional_leakage":
            # Directional leakage legend
            blue_patch = mpatches.Patch(
                color="blue", alpha=0.6, label="Forward Similarity"
            )
            orange_patch = mpatches.Patch(
                color="orange", alpha=0.6, label="Backward Similarity"
            )
            fig.legend(
                handles=[blue_patch, orange_patch],
                loc="lower center",
                bbox_to_anchor=(0.5, legend_y),
                ncol=2,
                fontsize=10,
                frameon=False,
                columnspacing=2.5,
                handletextpad=1.5,
            )

    # Create pooling strategy legend text (only if show_segment_lengths is True)
    if show_segment_lengths:
        # Move the legend box further down to avoid overlap with the similarity legend
        if single_model_mode:
            y_pos = -0.03  # Adjusted for new layout
        else:
            y_pos = -0.07  # Lowered from 0 to -0.08

        pooling_legend_lines = ["Pooling Strategies:"]
        if pooling_legend_type == "segment_standalone":
            pooling_legend_lines.append(
                f"Segment Standalone: {pooling_strategy_segment_standalone.upper()}"
            )
        elif pooling_legend_type == "segment_standalone_and_document":
            pooling_legend_lines.append(
                f"Segment Standalone: {pooling_strategy_segment_standalone.upper()}"
            )
            pooling_legend_lines.append(
                f"Document-Level: {pooling_strategy_document.upper()}"
            )
        pooling_legend_text = "\n".join(pooling_legend_lines)

        # Add ranges and pooling strategy legend in a single box if there are different ranges
        if ranges_info:
            # Build segment lengths legend
            legend_lines = ["Segment Lengths:"]
            for ranges_tuple, range_id in ranges_info.items():
                ranges_str = " | ".join(
                    [f"{start}-{end}" for start, end in ranges_tuple]
                )
                legend_lines.append(f"{range_id}=({ranges_str})")
            # Force correct label for the first line (avoid LaTeX/spacing issues)
            segment_lengths_legend_text = (
                r"$\mathbf{Segment\ Lengths:}$" + "\n" + "\n".join(legend_lines[1:])
            )

            # Pooling legend: only first line bold, force correct label
            pooling_legend_lines = pooling_legend_text.split("\n")
            pooling_legend_lines[0] = "Pooling Strategies:"
            pooling_legend_text = (
                r"$\mathbf{Pooling\ Strategies:}$"
                + "\n"
                + "\n".join(pooling_legend_lines[1:])
            )

            # Combine both legends horizontally with spacing
            combined_legend_text = (
                segment_lengths_legend_text + "\n\n" + pooling_legend_text
            )
            # Use a single box, with left alignment
            fig.text(
                0.5,  # Centered horizontally
                y_pos,
                combined_legend_text,
                ha="center",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.5, boxstyle="round,pad=0.7"),
                usetex=False,
            )
        else:
            # Only show pooling strategies legend centered
            pooling_legend_lines = pooling_legend_text.split("\n")
            pooling_legend_lines[0] = "Pooling Strategies:"
            pooling_legend_text = (
                r"$\mathbf{Pooling\ Strategies:}$"
                + "\n"
                + "\n".join(pooling_legend_lines[1:])
            )
            fig.text(
                0.5,  # Center position for the pooling strategies legend
                y_pos,
                pooling_legend_text,
                ha="left",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.5, boxstyle="round,pad=0.5"),
                usetex=False,
            )

    # For directional leakage, print comparison statistics
    if analysis_type == "directional_leakage" and False:
        print("\nStatistical Comparison of Forward vs Backward Similarity:")
        for result in all_results:
            path_name = result.get("path_name", "Unknown")
            print(f"{path_name}")
            print(f"  Forward Mean: {result['forward_mean']:.4f}")
            print(f"  Backward Mean: {result['backward_mean']:.4f}")
            print(
                f"  Difference (F-B): {result['forward_mean'] - result['backward_mean']:.4f}"
            )

            # Report t-test
            t_stat = result.get("t_stat", 0)
            p_value = result.get("p_value", 1.0)
            significance = "significant" if p_value < 0.05 else "not significant"
            print(f"  T-test: t={t_stat:.3f}, p={p_value:.6f} ({significance})\n")

    # Save plot if requested
    if save_plot:
        if save_path is None:
            filename = f"multi_{analysis_type}_analysis"
            if analysis_type == "position":
                filename += f"_{pooling_strategy_segment_standalone}_{pooling_strategy_document}"
            filename += ".png"
            save_path = os.path.join(str(paths[0]), filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()

    if return_full_results:
        return {"model_results": dict(model_groups)}

    return None


class MultiModelPositionSimilaritySinglePlotter:
    """
    A single plotter that shows multiple models as different lines instead of matryoshka dimensions.
    This works by creating synthetic "matryoshka_results" containing results from different models.
    """

    def __init__(self, model_pooling_strats: Dict[str, str], paths: List[str | Path]):
        self.model_pooling_strats = model_pooling_strats
        self.paths = paths
        # Group paths by experiment configuration
        self.config_to_paths = self._group_paths_by_config()

    def _group_paths_by_config(self):
        """Group paths by experiment configuration (excluding model name)."""
        from ..analysis.segment_embedding_analysis import load_exp_info

        config_groups = defaultdict(list)
        for path in self.paths:
            exp_info = load_exp_info(path)
            # Use source_lang and target_lang from the experiment info for proper language identification
            source_lang = exp_info.get("source_lang", "unknown")
            target_lang = exp_info.get("target_lang", None)

            # Create language key that distinguishes between monolingual and multilingual experiments
            if target_lang is None:
                lang_key = source_lang  # Monolingual: e.g., "de"
            else:
                lang_key = f"{source_lang}_{target_lang}"  # Multilingual: e.g., "de_en"

            config_key = (exp_info["concat_size"], lang_key)
            config_groups[config_key].append((path, exp_info["model_name"]))

        return config_groups

    def plot_position_similarities_in_subplot_with_precomputed(
        self,
        ax: plt.Axes,
        base_result: Dict[str, Any],
        all_model_results: Dict[Tuple, Dict[str, Any]],
        config_groups: Dict[Tuple, List[Tuple]],
        model_pooling_strats: Dict[str, str],
        latechunk_overlay_results: Optional[
            Dict[Tuple[int, str], Dict[str, Any]]
        ] = None,
        latechunk_overlay_label: str = "jina-v3",
        latechunk_overlay_color: str = "lightsalmon",
        show_title: bool = True,
        compact: bool = True,
        ylim: Optional[Tuple[float, float]] = None,
        show_segment_lengths: bool = False,
        show_lengths: bool = False,
        token_ylim: Optional[Tuple[float, float]] = None,
        show_token_ylabel: bool = True,
        show_token_ticklabels: bool = True,
        show_cosine_ticklabels: bool = True,
    ) -> None:
        """
        Plot position-based similarity results showing different models as different lines.
        Uses pre-computed results for proper y-axis scaling.
        """
        # Extract basic information
        position_means = base_result["position_means"]
        position_ci_lower = base_result["position_ci_lower"]
        position_ci_upper = base_result["position_ci_upper"]
        positions = list(range(1, len(position_means) + 1))

        # Extract range ID if available
        range_id = base_result.get("range_id", "N/A")

        # Find the config key for this result
        concat_size = base_result["concat_size"]
        source_lang = base_result.get("source_lang", "unknown")
        target_lang = base_result.get("target_lang", None)

        # Create language key that matches the grouping logic
        if target_lang is None:
            lang_key = source_lang  # Monolingual: e.g., "de"
        else:
            lang_key = f"{source_lang}_{target_lang}"  # Multilingual: e.g., "de_en"

        config_key = (concat_size, lang_key)

        # Define colors for different models
        model_colors = {
            "mGTE": "blue",
            "jina-v3": "red",
            # "qwen3-0.6B": "green"
        }

        # Get all models for this config and plot them
        if config_key in config_groups:
            for path, model_name in config_groups[config_key]:
                # Get abbreviated model name
                if "Alibaba-NLP/gte-multilingual-base" in model_name:
                    abbreviated_name = "mGTE"
                elif "jinaai/jina-embeddings-v3" in model_name:
                    abbreviated_name = "jina-v3"
                elif "Qwen/Qwen3-Embedding-0.6B" in model_name:
                    abbreviated_name = "qwen3-0.6B"
                else:
                    abbreviated_name = model_name

                # Get pre-computed result
                result_key = (config_key, model_name)
                if result_key in all_model_results:
                    result = all_model_results[result_key]
                    color = model_colors.get(abbreviated_name, "black")

                    # Plot this model's results
                    ax.plot(
                        positions,
                        result["position_means"],
                        "o-",
                        color=color,
                        linewidth=PLOT_LINEWIDTH,
                        markersize=PLOT_MARKERSIZE,
                        label=abbreviated_name,
                    )

                    # Add 95% confidence interval
                    for i, (pos, mean, ci_lower, ci_upper) in enumerate(
                        zip(
                            positions,
                            result["position_means"],
                            result["position_ci_lower"],
                            result["position_ci_upper"],
                        )
                    ):
                        bar_width = 0.4  # Same width as original single plotter
                        rect = plt.Rectangle(
                            (pos - bar_width / 2, ci_lower),
                            bar_width,
                            ci_upper - ci_lower,
                            color=color,
                            alpha=0.2,
                        )
                        ax.add_patch(rect)

        if (
            latechunk_overlay_results is not None
            and config_key in latechunk_overlay_results
        ):
            overlay = latechunk_overlay_results[config_key]
            assert len(overlay["position_means"]) == len(
                positions
            ), "Latechunk overlay must match base positions length."
            ax.plot(
                positions,
                overlay["position_means"],
                "o-",
                color=latechunk_overlay_color,
                linewidth=PLOT_LINEWIDTH,
                markersize=PLOT_MARKERSIZE,
                label=latechunk_overlay_label,
            )
            for pos, ci_lower, ci_upper in zip(
                positions,
                overlay["position_ci_lower"],
                overlay["position_ci_upper"],
            ):
                rect = plt.Rectangle(
                    (pos - 0.4 / 2, ci_lower),
                    0.4,
                    ci_upper - ci_lower,
                    color=latechunk_overlay_color,
                    alpha=0.2,
                )
                ax.add_patch(rect)

        # Add labels (compact for subplots)
        ax.set_xlabel(
            "Position",
            fontsize=BASE_AXIS_LABEL_FONT_SIZE * SUBPLOT_FONT_SCALE,
            labelpad=15,
        )
        ax.set_ylabel(
            "Cosine Similarity",
            fontsize=BASE_AXIS_LABEL_FONT_SIZE * SUBPLOT_FONT_SCALE,
        )

        # Add title with language information (same format as original plot method)
        if show_title:
            # Get language information if available
            source_lang = base_result.get("source_lang")
            target_lang = base_result.get("target_lang")

            # Add language information if available
            if target_lang and source_lang:
                title_text = (
                    f"Lang.: [{target_lang}, {source_lang}, ..., {source_lang}]"
                )
            elif source_lang:
                title_text = f"Lang.: [{source_lang}, ..., {source_lang}]"
            else:
                title_text = f"Concat Size: {concat_size}"

            if show_segment_lengths and range_id != "N/A":
                title_text += f"; SL:: {range_id}"

            ax.set_title(
                title_text,
                fontsize=BASE_SUBPLOT_TITLE_FONT_SIZE * SUBPLOT_FONT_SCALE,
            )

        # Set x-axis ticks to be integers
        ax.set_xticks(positions)

        # Set y-axis limits if provided
        if ylim is not None:
            ax.set_ylim((ylim[0], max(ylim[1], 1.0)))
        else:
            bottom, top = ax.get_ylim()
            if top < 1.0:
                ax.set_ylim((bottom, 1.0))

        ax.set_yticks([0.4, 0.6, 0.8, 1.0])
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))

        # Format y-axis to show appropriate precision and prevent duplicate labels
        # Use more decimal places if the y-range is small to avoid duplicate tick labels
        if ylim is not None:
            y_range = ylim[1] - ylim[0]
            if y_range < 0.3:
                # For small ranges, use 2 decimal places
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            else:
                # For larger ranges, 1 decimal place is sufficient
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        else:
            # Default formatting
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(
            axis="x",
            labelsize=BASE_TICK_LABEL_FONT_SIZE * SUBPLOT_FONT_SCALE,
            pad=POS_LABEL_PAD,
        )
        ax.tick_params(
            axis="y",
            labelleft=show_cosine_ticklabels,
            labelsize=BASE_TICK_LABEL_FONT_SIZE * SUBPLOT_FONT_SCALE,
        )

        # Add token length bar chart on right y-axis if requested
        if show_lengths and "token_lengths" in base_result:
            ax2 = ax.twinx()
            token_data = base_result["token_lengths"]
            token_means = token_data["position_means"]

            # Plot token length bars with transparency
            bars = ax2.bar(
                positions,
                token_means,
                alpha=0.3,
                color="gray",
                width=0.6,
                label="Token Lengths",
                zorder=1,  # Put bars behind line plots
            )

            # Set token length y-axis limits if provided
            if token_ylim is not None:
                ax2.set_ylim(token_ylim)

            # Format token length y-axis
            if show_token_ylabel:
                ax2.set_ylabel(
                    "Token Length",
                    color="gray",
                    fontsize=BASE_AXIS_LABEL_FONT_SIZE * SUBPLOT_FONT_SCALE,
                )
            ax2.tick_params(
                axis="y",
                labelcolor="gray",
                labelright=show_token_ticklabels,
                labelleft=False,
                labelsize=BASE_TICK_LABEL_FONT_SIZE * SUBPLOT_FONT_SCALE,
            )
            if not show_token_ticklabels:
                ax2.set_ylabel("")
            ax2.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

            # Ensure main plot is in front
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)  # Make main plot background transparent

        # Add grid for better readability
        ax.grid(True, which="major", linestyle="--", alpha=0.7)
        ax.grid(True, which="minor", linestyle="--", alpha=0.7)

    def plot_position_similarities_in_subplot(
        self,
        ax: plt.Axes,
        base_result: Dict[str, Any],
        show_title: bool = True,
        compact: bool = True,
        ylim: Optional[Tuple[float, float]] = None,
        show_segment_lengths: bool = False,
        show_lengths: bool = False,
        token_ylim: Optional[Tuple[float, float]] = None,
        show_token_ylabel: bool = True,
        show_token_ticklabels: bool = True,
        show_cosine_ticklabels: bool = True,
    ) -> None:
        """Fallback plotting path using only the provided base result."""

        concat_size = base_result["concat_size"]
        source_lang = base_result.get("source_lang", "unknown")
        target_lang = base_result.get("target_lang")
        lang_key = (
            source_lang if target_lang is None else f"{source_lang}_{target_lang}"
        )
        config_key = (concat_size, lang_key)
        model_name = base_result.get("model_name", "unknown")

        minimal_all_results: Dict[Tuple, Dict[str, Any]] = {
            (config_key, model_name): base_result
        }
        minimal_config_groups: Dict[Tuple, List[Tuple]] = defaultdict(list)
        path = base_result.get("path")
        assert path is not None, "Base result must include the experiment path."
        minimal_config_groups[config_key].append((path, model_name))

        self.plot_position_similarities_in_subplot_with_precomputed(
            ax,
            base_result,
            minimal_all_results,
            minimal_config_groups,
            self.model_pooling_strats,
            show_title=show_title,
            compact=compact,
            ylim=ylim,
            show_segment_lengths=show_segment_lengths,
            show_lengths=show_lengths,
            token_ylim=token_ylim,
            show_token_ylabel=show_token_ylabel,
            show_token_ticklabels=show_token_ticklabels,
            show_cosine_ticklabels=show_cosine_ticklabels,
        )
