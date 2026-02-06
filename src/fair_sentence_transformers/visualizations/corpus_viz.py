from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter


LengthThreshold = Mapping[str, Tuple[Optional[int], Optional[int]]]


def assert_shape(x: np.ndarray, shape: Tuple[int, ...]) -> None:
    assert tuple(x.shape) == tuple(
        shape
    ), f"Shape mismatch: got {tuple(x.shape)} expected {tuple(shape)}"


@dataclass(frozen=True)
class CorpusMetadata:
    long: pl.DataFrame
    wide: pl.DataFrame
    languages: Tuple[str, ...]

    def filter_by_length(
        self,
        thresholds: Optional[LengthThreshold] = None,
        relative_thresholds: Optional[Mapping[str, float]] = None,
        reference_language: str = "en",
    ) -> "CorpusMetadata":
        if thresholds is None:
            thresholds = {}
        if relative_thresholds is None:
            relative_thresholds = {}

        assert reference_language in self.languages, "Reference language missing"

        unexpected = set(thresholds) - set(self.languages)
        assert (
            not unexpected
        ), f"Thresholds provided for unknown languages: {sorted(unexpected)}"

        unexpected_relative = set(relative_thresholds) - set(self.languages)
        assert (
            not unexpected_relative
        ), f"Relative thresholds provided for unknown languages: {sorted(unexpected_relative)}"

        filtered_wide = self.wide
        for language in self.languages:
            if language not in thresholds:
                continue
            lower, upper = thresholds[language]
            if lower is not None:
                filtered_wide = filtered_wide.filter(pl.col(language) >= int(lower))
            if upper is not None:
                filtered_wide = filtered_wide.filter(pl.col(language) <= int(upper))

        if relative_thresholds:
            mask_expr = pl.lit(True)
            ref_col = pl.col(reference_language)
            for language, tolerance in relative_thresholds.items():
                tol = float(tolerance)
                assert (
                    tol >= 0.0
                ), f"Relative threshold must be non-negative for {language}"
                lower_bound = ref_col * (1.0 - tol)
                upper_bound = ref_col * (1.0 + tol)
                condition = (pl.col(language) >= lower_bound) & (
                    pl.col(language) <= upper_bound
                )
                mask_expr = mask_expr & condition
            filtered_wide = filtered_wide.filter(mask_expr)

        filtered_long = self.long.join(
            filtered_wide.select("dataset_idx"), on="dataset_idx", how="inner"
        ).sort(["language", "dataset_idx"])
        return CorpusMetadata(
            long=filtered_long, wide=filtered_wide, languages=self.languages
        )


def load_corpus_metadata(config_path: Path | str) -> CorpusMetadata:
    config_file = Path(config_path)
    assert config_file.is_file(), f"Config file does not exist: {config_file}"

    with config_file.open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    metadata_paths = config.get("metadata_path")
    assert isinstance(metadata_paths, Mapping), "metadata_path must be a mapping"

    records: list[pl.DataFrame] = []
    languages: list[str] = []
    for language, path_str in metadata_paths.items():
        path = Path(path_str)
        assert path.is_file(), f"Metadata file missing for {language}: {path}"
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        assert (
            isinstance(payload, Mapping) and payload
        ), f"Metadata payload must be a non-empty mapping for {language}"

        lang_records = []
        for value in payload.values():
            assert (
                value["language"] == language
            ), f"Language mismatch in {language} metadata"
            lang_records.append(
                {
                    "dataset_idx": int(value["dataset_idx"]),
                    "token_length": (
                        int(value["token_length"]) + 2
                    ),  # account for special tokens
                    "topic": value["topic"],
                    "pair_id": value["pair_id"],
                    "language": language,
                }
            )

        lang_df = pl.DataFrame(lang_records).sort("dataset_idx")
        assert (
            lang_df["dataset_idx"].is_unique().all()
        ), f"Duplicate dataset_idx entries for {language}"
        assert lang_df.height > 0, f"No entries for language {language}"
        records.append(lang_df)
        languages.append(language)

    sorted_languages = tuple(languages)
    long_df = pl.concat(records).sort(["language", "dataset_idx"])

    wide_df = (
        long_df.select(["dataset_idx", "language", "token_length"])
        .pivot(values="token_length", index="dataset_idx", on="language")
        .sort("dataset_idx")
    )
    assert wide_df.height > 0, "Wide metadata frame must not be empty"

    expected_columns = {"dataset_idx", *sorted_languages}
    assert set(wide_df.columns) == expected_columns, "Wide frame columns mismatch"

    return CorpusMetadata(long=long_df, wide=wide_df, languages=sorted_languages)


def compute_language_length_summary(corpus: CorpusMetadata) -> pl.DataFrame:
    return (
        corpus.long.group_by("language")
        .agg(
            [
                pl.len().alias("count"),
                pl.col("token_length").mean().alias("mean_length"),
                pl.col("token_length").median().alias("median_length"),
                pl.col("token_length").std().alias("std_length"),
                pl.col("token_length").min().alias("min_length"),
                pl.col("token_length").max().alias("max_length"),
            ]
        )
        .sort("language")
    )


def compute_instance_length_mad(corpus: CorpusMetadata) -> pl.DataFrame:
    length_array = corpus.wide.select(corpus.languages).to_numpy()
    assert_shape(length_array, (corpus.wide.height, len(corpus.languages)))

    means = length_array.mean(axis=1, keepdims=True)
    assert_shape(means, (corpus.wide.height, 1))
    mad = np.abs(length_array - means).mean(axis=1)
    assert_shape(mad, (corpus.wide.height,))

    return pl.DataFrame(
        {
            "dataset_idx": corpus.wide["dataset_idx"].to_numpy(),
            "mad_token_length": mad,
        }
    )


def plot_token_length_distributions(
    corpus: CorpusMetadata,
    bins: int = 50,
    palette: Optional[Sequence[str]] = None,
) -> Figure:
    languages = corpus.languages
    n_langs = len(languages)
    assert n_langs > 0, "No languages available for plotting"

    cols = min(3, n_langs)
    rows = math.ceil(n_langs / cols)
    width = 5.0 * cols
    height = 3.0 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(width, height))
    axes_arr = np.array(axes).reshape(rows, cols)

    sns.set_style("whitegrid")
    if palette is None:
        palette = sns.color_palette("viridis", n_colors=n_langs)

    for idx, language in enumerate(languages):
        row = idx // cols
        col = idx % cols
        ax = axes_arr[row, col]
        language_df = corpus.long.filter(pl.col("language") == language)
        token_lengths = language_df["token_length"].to_numpy()
        assert_shape(token_lengths, (language_df.height,))
        sns.histplot(token_lengths, bins=bins, ax=ax, color=palette[idx], kde=False)
        ax.set_title(language)
        ax.set_xlabel("Token length")
        ax.set_ylabel("Count")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    for idx in range(n_langs, rows * cols):
        row = idx // cols
        col = idx % cols
        axes_arr[row, col].axis("off")

    fig.tight_layout()
    return fig


def plot_instance_length_mad_distribution(
    corpus: CorpusMetadata,
    bins: int = 50,
    color: Optional[str] = None,
) -> Figure:
    mad_df = compute_instance_length_mad(corpus)
    values = mad_df["mad_token_length"].to_numpy()
    assert_shape(values, (mad_df.height,))

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(5.0, 3.0))
    ax = fig.add_subplot(1, 1, 1)
    sns.histplot(
        values,
        bins=bins,
        color=color or sns.color_palette("viridis", 1)[0],
        ax=ax,
        kde=False,
    )
    ax.set_title("Mean absolute deviation of token lengths")
    ax.set_xlabel("Average deviation from per-instance mean")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    fig.tight_layout()
    return fig


def build_corpus_length_summary_table(summary: pl.DataFrame) -> str:
    assert isinstance(summary, pl.DataFrame) and summary.height > 0
    expected_cols = {"language", "mean_length", "median_length", "count"}
    assert expected_cols.issubset(set(summary.columns))
    iso2lang = {
        "en": "English",
        "zh": "Chinese",
        "de": "German",
        "it": "Italian",
        "ko": "Korean",
        "hi": "Hindi",
    }
    desired_order = ["en", "zh", "de", "it", "ko", "hi"]

    summary_map = {str(row["language"]): row for row in summary.iter_rows(named=True)}
    unexpected = set(summary_map) - set(iso2lang)
    assert not unexpected, f"Unexpected languages present: {sorted(unexpected)}"

    present_order = [iso for iso in desired_order if iso in summary_map]
    assert present_order, "Summary must contain at least one supported language"

    rows: list[str] = []
    for idx, iso in enumerate(present_order):
        row = summary_map[iso]
        mean_val = int(round(float(row["mean_length"])))
        median_val = int(round(float(row["median_length"])))
        count_val = int(row["count"])
        display_name = iso2lang[iso]
        rows.append(
            f"    \\textbf{{{display_name}}} & {mean_val} & {median_val} & {count_val} \\\\"
        )
        if idx != len(present_order) - 1:
            rows.append("    \\addlinespace[2ex]")

    lines = [
        "\\begin{table}[tb]",
        "  \\centering",
        "  \\small",
        "  \\setlength{\\tabcolsep}{6pt}",
        "  \\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}} l *{3}{c} @{} }",
        "    \\multicolumn{4}{c}{\\textbf{Statistics of Wikipedia Comparable Corpus}} \\\\",
        "    \\addlinespace[2ex]",
        "    \\toprule",
        "    \\addlinespace[2ex]",
        "    & Token Length (Mean) & Token Length (Median) & Count \\\\",
        "    \\midrule",
    ]
    lines.extend(rows)
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular*}",
            "  \\caption[Statistics Wikipedia Comparable Corpus]{Statistics of the Wikipedia comparable corpus across six languages. Each row provides summary statistics for a given language. The token lengths are based on the XLM-R tokenizer. Note the relatively low count of articles due to the strict length filtering applied to increase comparability (cf. \\cref{subsubsec:quasi-parallel-construction}).}",
            "  \\label{tab:wiki-comparable-stats}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "CorpusMetadata",
    "LengthThreshold",
    "assert_shape",
    "load_corpus_metadata",
    "compute_language_length_summary",
    "compute_instance_length_mad",
    "plot_token_length_distributions",
    "plot_instance_length_mad_distribution",
    "build_corpus_length_summary_table",
]
