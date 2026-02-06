import os

from typing import Dict, List, Tuple, Set, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from pathlib import Path
import numpy as np

from .plot_constants import (
    SUBPLOT_FONT_SCALE,
    BASE_AXIS_LABEL_FONT_SIZE,
    BASE_TICK_LABEL_FONT_SIZE,
    BASE_SUBPLOT_TITLE_FONT_SIZE,
    PLOT_LINEWIDTH,
    PLOT_MARKERSIZE,
    POS_LABEL_PAD,
)

# Import the required analyses functions
from ..analysis.segment_embedding_analysis import (
    DocumentSegmentSimilarityAnalyzer,
    DirectionalLeakageAnalyzer,
    PositionalDirectionalLeakageAnalyzer,
)


class PositionSimilaritySinglePlotter:
    """
    Class for plotting position-based similarity results in a single plot.
    """

    def __init__(self):
        pass

    def plot_position_similarities_in_subplot(
        self,
        ax: plt.Axes,
        results: Dict[str, Any],
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
        Plot position-based similarity results in a given subplot.

        Args:
            ax: Matplotlib Axes object to plot on
            results: Dictionary returned from run_position_analysis()
            show_title: Whether to show the title (default: True)
            compact: Whether to use a compact plot style for multi-plot figures (default: True)
            ylim: Optional tuple specifying (min, max) y-axis limits
            show_segment_lengths: Whether to show segment length information in title and legends (default: True)
            show_lengths: Whether to show token lengths as bar charts on right y-axis (default: False)
            token_ylim: Optional tuple specifying (min, max) token length y-axis limits
            show_token_ylabel: Whether to show the "Token Length" label on the right y-axis (default: True)
        """
        # Extract data from results
        position_means = results["position_means"]
        position_ci_lower = results["position_ci_lower"]
        position_ci_upper = results["position_ci_upper"]
        positions = list(
            range(1, len(position_means) + 1)
        )  # 1-based positions for x-axis

        # Extract range ID if available
        range_id = results.get("range_id", "N/A")

        # Use abbreviated model name if available
        model_name = results.get(
            "abbreviated_model_name", results.get("model_name", "Unknown model")
        )

        # Plot means with a line (full embeddings)
        ax.plot(
            positions,
            position_means,
            "o-",
            color="red",
            linewidth=PLOT_LINEWIDTH,
            markersize=PLOT_MARKERSIZE,
            label="Full",
        )

        # Add 95% confidence interval for full embeddings
        for i, (pos, mean, ci_lower, ci_upper) in enumerate(
            zip(positions, position_means, position_ci_lower, position_ci_upper)
        ):
            # Width of the confidence interval bar
            bar_width = 0.4
            # Create rectangle patch for the confidence interval
            rect = plt.Rectangle(
                (pos - bar_width / 2, ci_lower),  # (x, y) of bottom left corner
                bar_width,  # width
                ci_upper - ci_lower,  # height
                color="red",
                alpha=0.2,
            )
            ax.add_patch(rect)

        # Plot Matryoshka dimensions if available
        if "matryoshka_results" in results and "matryoshka_dimensions" in results:
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
            matryoshka_results = results["matryoshka_results"]
            matryoshka_dimensions = results["matryoshka_dimensions"]

            for i, dim in enumerate(matryoshka_dimensions):
                if dim in matryoshka_results:
                    color = colors[i % len(colors)]
                    dim_means = matryoshka_results[dim]["position_means"]
                    dim_ci_lower = matryoshka_results[dim]["position_ci_lower"]
                    dim_ci_upper = matryoshka_results[dim]["position_ci_upper"]

                    # Plot means for this dimension
                    ax.plot(
                        positions,
                        dim_means,
                        "o-",
                        color=color,
                        linewidth=PLOT_LINEWIDTH,
                        markersize=PLOT_MARKERSIZE,
                        label=f"D{dim}",
                        linestyle=(
                            "--" if i >= 4 else "-"
                        ),  # Use dashed lines for dimensions 5 and beyond
                    )

                    # Add confidence intervals for this dimension
                    for j, (pos, mean, ci_lower, ci_upper) in enumerate(
                        zip(positions, dim_means, dim_ci_lower, dim_ci_upper)
                    ):
                        bar_width = 0.3  # Slightly narrower for Matryoshka dimensions
                        rect = plt.Rectangle(
                            (pos - bar_width / 2, ci_lower),
                            bar_width,
                            ci_upper - ci_lower,
                            color=color,
                            alpha=0.15,  # More transparent for Matryoshka dimensions
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

        # Add segment length as subtitle instead of text in the plot
        if show_title:
            current_title = ax.get_title()

            # Get language information if available
            source_lang = results.get("source_lang")
            target_lang = results.get("target_lang")

            # Add language information if available
            if target_lang and source_lang:
                title_text = (
                    f"Lang.: [{target_lang}, {source_lang}, ..., {source_lang}]"
                )
            elif source_lang:
                title_text = f"Lang.: [{source_lang}, ..., {source_lang}]"

            if show_segment_lengths:
                title_text += f"; SL:: {range_id}"

            computed_title = (
                f"{current_title}\n{title_text}" if current_title else title_text
            )
            ax.set_title(
                computed_title,
                fontsize=BASE_SUBPLOT_TITLE_FONT_SIZE * SUBPLOT_FONT_SCALE,
            )

        # Set x-axis ticks to be integers
        ax.set_xticks(positions)

        ax.set_yticks([0.4, 0.6, 0.8, 1.0])

        # Set y-axis limits if provided
        if ylim is not None:
            ax.set_ylim((ylim[0], max(ylim[1], 1.0)))
        else:
            bottom, top = ax.get_ylim()
            if top < 1.0:
                ax.set_ylim((bottom, 1.0))

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
        if show_lengths and "token_lengths" in results:
            token_data = results["token_lengths"]
            token_means = token_data["position_means"]

            # Create secondary y-axis for token lengths
            ax2 = ax.twinx()

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
        ax.grid(True, linestyle="--", alpha=0.7)

        # Note: Individual subplot legends are not shown - main legend is at the bottom of the figure


class DirectionalLeakageSinglePlotter:
    """
    Class to handle plotting of directional leakage results in a single plot.
    """

    def __init__(self):
        pass

    def plot_directional_leakage_in_subplot(
        self,
        ax: plt.Axes,
        results: Dict[str, Any],
        show_title: bool = True,
        compact: bool = True,
        xlim: Optional[Tuple[float, float]] = None,
        show_segment_lengths: bool = False,
    ) -> None:
        """
        Plot directional leakage results in an existing subplot.

        Args:
            ax: Matplotlib axes to plot on
            results: Results from run_directional_leakage_analysis
            show_title: Whether to show the title
            compact: Whether to use a compact plot style for multi-plot figures
            xlim: Optional tuple specifying (min, max) x-axis limits
        """
        # Create histograms comparing forward and backward influence
        num_bins = 20 if compact else 30
        alpha = 0.6 if compact else 0.7
        num_bins = 30
        alpha = 0.7

        # Plot histograms
        ax.hist(
            results["forward_influence"],
            alpha=alpha,
            label=f"F: {results['forward_mean']:.3f}" if compact else "Forward",
            bins=num_bins,
            color="blue",
        )
        ax.hist(
            results["backward_influence"],
            alpha=alpha,
            label=f"B: {results['backward_mean']:.3f}" if compact else "Backward",
            bins=num_bins,
            color="orange",
        )

        # Add mean lines
        ax.axvline(
            x=results["forward_mean"],
            color="blue",
            linestyle="--",
            label=None if compact else f"Forward Mean: {results['forward_mean']:.4f}",
        )
        ax.axvline(
            x=results["backward_mean"],
            color="orange",
            linestyle="--",
            label=None if compact else f"Backward Mean: {results['backward_mean']:.4f}",
        )

        # Extract range ID if available
        range_id = results.get("range_id", "N/A")

        if show_title:
            # Extract model name and size info
            model_name = results["model_name"]
            # Abbreviate model names
            if "Alibaba-NLP/gte-multilingual-base" in model_name:
                model_name = "mGTE"
            elif "jinaai/jina-embeddings-v3" in model_name:
                model_name = "jina-v3"
            elif "Qwen/Qwen3-Embedding-0.6B" in model_name:
                model_name = "qwen3-0.6B"
            # Store result with abbreviated model name
            results["abbreviated_model_name"] = model_name

            # Always add segment length as a subtitle
            if compact:
                # For multi-plot displays, show the segment length and language info if available
                # Get language information if available
                source_lang = results.get("source_lang")
                target_lang = results.get("target_lang")

                # Add language information if available
                if target_lang and source_lang:
                    title_text = (
                        f"Lang.: [{target_lang}, {source_lang}, ..., {source_lang}]"
                    )
                elif source_lang:
                    title_text = f"Lang.: [{source_lang}, ..., {source_lang}]"

                if show_segment_lengths:
                    title_text += f"; SL:: {range_id}"

                ax.set_title(
                    title_text,
                    fontsize=BASE_SUBPLOT_TITLE_FONT_SIZE * SUBPLOT_FONT_SCALE,
                )
            else:
                # For single plots, show more detailed information
                size_info = str(results["concat_size"])
                # Extract ranges information
                ranges = []
                if results["position_specific_ranges"]:
                    for start, end in results["position_specific_ranges"]:
                        ranges.append(f"{start}-{end}")
                ranges_str = " | ".join(ranges) if ranges else "N/A"

                # Get language information if available
                source_lang = results.get("source_lang")
                target_lang = results.get("target_lang")

                title_text = f"{model_name} (size {size_info})\nSL: {range_id}"

                # Add language information if available
                if target_lang and source_lang:
                    title_text += f"; Lang:: f: {target_lang}, re: {source_lang}"
                elif source_lang:
                    title_text += f"; Lang:: {source_lang}"

                ax.set_title(
                    title_text,
                    fontsize=(BASE_SUBPLOT_TITLE_FONT_SIZE + 1) * SUBPLOT_FONT_SCALE,
                )

        # Always set axis labels, even in compact mode
        ax.set_xlabel(
            "Cosine Similarity",
            fontsize=BASE_AXIS_LABEL_FONT_SIZE * SUBPLOT_FONT_SCALE,
        )
        ax.set_ylabel("Count", fontsize=BASE_AXIS_LABEL_FONT_SIZE * SUBPLOT_FONT_SCALE)

        ax.tick_params(
            axis="x", labelsize=BASE_TICK_LABEL_FONT_SIZE * SUBPLOT_FONT_SCALE
        )
        ax.tick_params(
            axis="y",
            labelsize=BASE_TICK_LABEL_FONT_SIZE * SUBPLOT_FONT_SCALE,
        )

        # Add legend
        ax.legend(fontsize=8 if compact else 10)

        # Set x-axis limits
        if xlim is not None:
            # Use provided global limits for consistent scaling across model
            ax.set_xlim(xlim)
        elif (
            len(results["forward_influence"]) > 0
            and len(results["backward_influence"]) > 0
        ):
            # Fallback to automatic limits if no global limits provided
            all_values = results["forward_influence"] + results["backward_influence"]
            min_val = max(min(all_values) - 0.01, -1.0)  # Cosine sim range is [-1, 1]
            max_val = min(max(all_values) + 0.01, 1.0)
            ax.set_xlim(min_val, max_val)


class PositionalDirectionalLeakageSinglePlotter:
    """
    Class for plotting positional directional leakage results in a single plot.
    """

    def __init__(self):
        pass

    def plot_positional_directional_leakage_in_subplot(
        self,
        ax: plt.Axes,
        results: Dict[str, Any],
        show_title: bool = True,
        compact: bool = True,
        ylim: Optional[Tuple[float, float]] = None,
        show_segment_lengths: bool = False,
    ) -> None:
        """
        Plot positional directional leakage results in a given subplot.

        Args:
            ax: Matplotlib Axes object to plot on
            results: Dictionary returned from run_positional_directional_leakage_analysis()
            show_title: Whether to show the title (default: True)
            compact: Whether to use a compact plot style for multi-plot figures (default: True)
            ylim: Optional tuple specifying (min, max) y-axis limits
            show_segment_lengths: Whether to show segment length information in title and legends (default: True)
        """
        # Extract data from results
        position_forward_means = results["position_forward_means"]
        position_backward_means = results["position_backward_means"]
        all_positions = results["all_positions"]

        # Convert to lists for plotting, ensuring 1-based indexing
        positions = list(range(1, len(all_positions) + 1))
        forward_means = [position_forward_means.get(pos, 0) for pos in all_positions]
        backward_means = [
            position_backward_means.get(pos, 0) for pos in all_positions
        ]  # Plot forward influence (excluding last position which is always 0)
        forward_positions = positions[:-1] if len(positions) > 1 else positions
        forward_values = forward_means[:-1] if len(forward_means) > 1 else forward_means

        if forward_positions:
            ax.plot(
                forward_positions,
                forward_values,
                "o",
                color="blue",
                linewidth=2,
                linestyle="-",
                # label="Forward",
                markersize=5,
            )

        # Plot backward influence (excluding first position which is always 0)
        backward_positions = positions[1:] if len(positions) > 1 else []
        backward_values = backward_means[1:] if len(backward_means) > 1 else []

        if backward_positions:
            ax.plot(
                backward_positions,
                backward_values,
                "o",
                color="orange",
                linewidth=2,
                linestyle="-",
                # label="Backward",
                markersize=5,
            )

        # Plot Matryoshka dimensions if available
        if "matryoshka_results" in results and "matryoshka_dimensions" in results:
            colors = [
                "red",
                "green",
                "purple",
                "brown",
                "pink",
                "gray",
                "olive",
            ]
            matryoshka_results = results["matryoshka_results"]
            matryoshka_dimensions = results["matryoshka_dimensions"]

            for i, dim in enumerate(matryoshka_dimensions):
                if dim in matryoshka_results:
                    color = colors[i % len(colors)]
                    dim_forward_means = matryoshka_results[dim][
                        "position_forward_means"
                    ]
                    dim_backward_means = matryoshka_results[dim][
                        "position_backward_means"
                    ]

                    # Convert to lists for plotting
                    dim_forward_values = [
                        dim_forward_means.get(pos, 0) for pos in all_positions
                    ]
                    dim_backward_values = [
                        dim_backward_means.get(pos, 0) for pos in all_positions
                    ]

                    # Plot forward influence for this dimension (excluding last position)
                    dim_forward_positions = forward_positions
                    dim_forward_vals = (
                        dim_forward_values[:-1]
                        if len(dim_forward_values) > 1
                        else dim_forward_values
                    )

                    if dim_forward_positions:
                        ax.plot(
                            dim_forward_positions,
                            dim_forward_vals,
                            "o",
                            color=color,
                            linewidth=2,
                            label=f"F D{dim}",
                            linestyle="--",
                            markersize=4,
                        )

                    # Plot backward influence for this dimension (excluding first position)
                    dim_backward_positions = backward_positions
                    dim_backward_vals = (
                        dim_backward_values[1:] if len(dim_backward_values) > 1 else []
                    )

                    if dim_backward_positions:
                        ax.plot(
                            dim_backward_positions,
                            dim_backward_vals,
                            "s",
                            color=color,
                            linewidth=2,
                            label=f"B D{dim}",
                            linestyle=":",
                            markersize=4,
                        )

        # Add average lines (excluding zeros)
        # Calculate average forward influence (excluding last position which is 0)
        if forward_values:
            forward_avg = np.mean(forward_values)
            ax.axhline(
                y=forward_avg, color="blue", linestyle="--", alpha=0.7, linewidth=1
            )

        # Calculate average backward influence (excluding first position which is 0)
        if backward_values:
            backward_avg = np.mean(backward_values)
            ax.axhline(
                y=backward_avg, color="orange", linestyle="--", alpha=0.7, linewidth=1
            )

        # Add labels
        ax.set_xlabel(
            "Position",
            fontsize=BASE_AXIS_LABEL_FONT_SIZE * SUBPLOT_FONT_SCALE,
            labelpad=15,
        )
        ax.set_ylabel(
            "Cosine Similarity",
            fontsize=BASE_AXIS_LABEL_FONT_SIZE * SUBPLOT_FONT_SCALE,
        )
        ax.tick_params(
            axis="x", labelsize=BASE_TICK_LABEL_FONT_SIZE * SUBPLOT_FONT_SCALE
        )
        ax.tick_params(
            axis="y",
            labelsize=BASE_TICK_LABEL_FONT_SIZE * SUBPLOT_FONT_SCALE,
        )

        # Set y-axis limits if provided
        if ylim:
            ax.set_ylim(ylim)

        # Add legend (compact for subplots)
        if compact:
            ax.legend(loc="best", fontsize="small")
        else:
            ax.legend(loc="best")

        # Extract range ID if available
        range_id = results.get("range_id", "N/A")

        # Use abbreviated model name if available
        model_name = results.get(
            "abbreviated_model_name", results.get("model_name", "Unknown model")
        )

        if show_title:
            current_title = ax.get_title()

            # Get language information if available
            source_lang = results.get("source_lang")
            target_lang = results.get("target_lang")

            # Add language information if available
            if target_lang and source_lang:
                title_text = (
                    f"Lang.: [{target_lang}, {source_lang}, ..., {source_lang}]"
                )
            elif source_lang:
                title_text = f"Lang.: [{source_lang}, ..., {source_lang}]"
            else:
                title_text = "Position-wise Influence on Segment Contextualization"

            if show_segment_lengths:
                # Add range information as subtitle
                title_text += f"; SL:: {range_id}"

            computed_title = (
                f"{current_title}\n{title_text}" if current_title else title_text
            )
            ax.set_title(
                computed_title,
                fontsize=BASE_SUBPLOT_TITLE_FONT_SIZE * SUBPLOT_FONT_SCALE,
            )

        # Set integer x-axis ticks
        ax.set_xticks(positions)

        # Grid for better readability
        ax.grid(True, alpha=0.3)


class PositionalDirectionalLeakageHeatmapSinglePlotter:
    """
    Class for plotting positional directional leakage results as heatmaps.
    Shows detailed position-wise similarities in matrix form.
    """

    def __init__(self):
        pass

    def plot_heatmap_in_subplot(
        self,
        ax: plt.Axes,
        results: Dict[str, Any],
        show_title: bool = True,
        compact: bool = True,
        show_segment_lengths: bool = False,
    ) -> None:
        """
        Plot positional directional leakage results as a heatmap in a given subplot.

        Args:
            ax: Matplotlib Axes object to plot on
            results: Dictionary returned from run_positional_directional_leakage_analysis()
            show_title: Whether to show the title (default: True)
            compact: Whether to use a compact plot style for multi-plot figures (default: True)
            show_segment_lengths: Whether to show segment length information in title and legends (default: True)
        """
        # Extract data from results
        position_forward_influence = results["position_forward_influence"]
        position_backward_influence = results["position_backward_influence"]
        all_positions = results["all_positions"]

        # Create position-wise similarity matrix
        # We'll create a matrix where each cell (i,j) represents the similarity
        # Upper triangle: forward similarities (pos i -> pos j, where j > i)
        # Lower triangle: backward similarities (pos i -> pos j, where i > j)
        max_pos = max(all_positions)
        similarity_matrix = np.full((max_pos, max_pos), np.nan)

        # For now, we use the available aggregated data to approximate the heatmap
        # This is a limitation of the current data structure, which aggregates similarities per position
        # For exact pairwise data, the core analysis would need modification

        # Fill upper triangle with forward similarities
        for i in range(max_pos):
            pos_i = i + 1  # Convert to 1-based position
            if pos_i in position_forward_influence:
                forward_values = position_forward_influence[pos_i]
                if forward_values:
                    avg_forward = np.mean(forward_values)
                    # Distribute this average to all forward positions
                    for j in range(i + 1, max_pos):
                        similarity_matrix[i, j] = avg_forward

        # Fill lower triangle with backward similarities
        for i in range(max_pos):
            pos_i = i + 1  # Convert to 1-based position
            if pos_i in position_backward_influence:
                backward_values = position_backward_influence[pos_i]
                if backward_values:
                    avg_backward = np.mean(backward_values)
                    # Distribute this average to all backward positions
                    for j in range(i):
                        similarity_matrix[i, j] = avg_backward

        # Create heatmap with a diverging colormap
        im = ax.imshow(
            similarity_matrix,
            cmap="RdBu_r",  # Red-Blue reversed (red for high similarity)
            aspect="equal",
            interpolation="nearest",
            vmin=-1,  # Cosine similarity range
            vmax=1,
        )

        # Set ticks and labels
        ax.set_xticks(range(max_pos))
        ax.set_yticks(range(max_pos))
        tick_fontsize = (8 if compact else 10) * SUBPLOT_FONT_SCALE
        ax.set_xticklabels([f"P{i+1}" for i in range(max_pos)], fontsize=tick_fontsize)
        ax.set_yticklabels([f"P{i+1}" for i in range(max_pos)], fontsize=tick_fontsize)

        # Add grid between cells
        ax.set_xticks(np.arange(-0.5, max_pos, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, max_pos, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1)

        # Add value annotations to cells
        for i in range(max_pos):
            for j in range(max_pos):
                if not np.isnan(similarity_matrix[i, j]):
                    value = similarity_matrix[i, j]
                    # Choose text color based on value for readability
                    text_color = "white" if abs(value) > 0.5 else "black"
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=(6 if compact else 8) * SUBPLOT_FONT_SCALE,
                    )

        # Set labels
        label_fontsize = (9 if compact else 11) * SUBPLOT_FONT_SCALE
        ax.set_xlabel("Target Position", fontsize=label_fontsize)
        ax.set_ylabel("Source Position", fontsize=label_fontsize)
        ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)

        # Add title
        if show_title:
            # Extract range ID if available
            range_id = results.get("range_id", "N/A")

            # Use abbreviated model name if available
            model_name = results.get(
                "abbreviated_model_name", results.get("model_name", "Unknown model")
            )

            # Get language information if available
            source_lang = results.get("source_lang")
            target_lang = results.get("target_lang")

            # Add language information if available
            if target_lang and source_lang:
                title_text = (
                    f"Lang.: [{target_lang}, {source_lang}, ..., {source_lang}]"
                )
            elif source_lang:
                title_text = f"Lang.: [{source_lang}, ..., {source_lang}]"
            else:
                title_text = "Position-wise Similarity Matrix"

            if show_segment_lengths:
                # Add range information as subtitle
                title_text += f"; SL:: {range_id}"

            title_fontsize = (11 if compact else 13) * SUBPLOT_FONT_SCALE
            ax.set_title(title_text, fontsize=title_fontsize)

        # Add legend text explaining the matrix layout
        legend_text = "Upper: Forward\nLower: Backward"
        ax.text(
            0.02,
            0.98,
            legend_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=(6 if compact else 8) * SUBPLOT_FONT_SCALE,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

        # Add a small colorbar if not in compact mode
        if not compact:
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax, label="Cosine Similarity")

    def create_detailed_similarity_matrix(self, results: Dict[str, Any]) -> np.ndarray:
        """
        Create a detailed similarity matrix with individual pairwise similarities.
        This function reconstructs the pairwise similarities from the raw data.

        Note: The current implementation uses averaged values due to data structure limitations.
        For exact pairwise similarities, the core analysis would need to be modified to
        store individual pairwise similarities rather than position-aggregated lists.

        Args:
            results: Dictionary returned from run_positional_directional_leakage_analysis()

        Returns:
            NumPy array representing the similarity matrix
        """
        position_forward_influence = results["position_forward_influence"]
        position_backward_influence = results["position_backward_influence"]
        all_positions = results["all_positions"]

        max_pos = max(all_positions)
        similarity_matrix = np.full((max_pos, max_pos), np.nan)

        # This is a simplified approach - we use the mean of all influences from each position
        # For more precision, the core analysis would need to track individual pairwise similarities
        for i, pos_i in enumerate(all_positions):
            # Forward influences from pos_i
            forward_influences = position_forward_influence.get(pos_i, [])
            if forward_influences:
                forward_mean = np.mean(forward_influences)
                for j, pos_j in enumerate(all_positions):
                    if pos_j > pos_i:
                        similarity_matrix[pos_i - 1, pos_j - 1] = forward_mean

            # Backward influences from pos_i
            backward_influences = position_backward_influence.get(pos_i, [])
            if backward_influences:
                backward_mean = np.mean(backward_influences)
                for j, pos_j in enumerate(all_positions):
                    if pos_i > pos_j:
                        similarity_matrix[pos_i - 1, pos_j - 1] = backward_mean

        return similarity_matrix
