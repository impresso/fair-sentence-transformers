"""
Numerical analysis utilities for fair_sentence_transformers.

This module provides functions to compute and return numerical results that are
otherwise used for plotting, enabling downstream statistical analysis without
rendering figures.
"""

from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
import pandas as pd  # Needed for long-format assembly for OLS
import statsmodels.api as sm  # For covariance access
import statsmodels.formula.api as smf  # Formula OLS
import pingouin as pg  # Repeated-measures ANOVA

# Reuse the analyzers and experiment info loader used by the visualization code
from .segment_embedding_analysis import (
    DocumentSegmentSimilarityAnalyzer,
    load_exp_info,
)


def collect_multi_model_position_analysis_results(
    paths: List[str | Path],
    model_pooling_strats: Dict[str, str],
    document_embedding_type: str = "document-level",
) -> Dict[Tuple[Tuple[int, str], str], Dict[str, Any]]:
    """
    Compute and collect position-analysis results for multiple experiments and models.

    This mirrors the precomputation part of
    DocumentLevel2SegmentStandaloneSimPlotter._analyze_and_plot_multiple_results_multi_model,
    up to and including the assignment:

            all_model_results[result_key] = result

    Inputs follow the same format as plot_multi_models():
    - paths: list of experiment directories
    - model_pooling_strats: mapping of model_name -> pooling strategy ("cls" or "mean")

    Returns:
            A dictionary keyed by ((concat_size, lang_key), model_name) where lang_key is either
            "<src>" (monolingual) or "<src>_<tgt>" (multilingual), and values are the result dicts
            returned by DocumentSegmentSimilarityAnalyzer.run_position_analysis().
    """

    if not paths:
        return {}

    # Setup analysis (kept as close as possible to existing visualization code)
    doc_seg_analyzer = DocumentSegmentSimilarityAnalyzer()

    # Group experiments by unique config (concat size + language), excluding model
    config_groups: Dict[Tuple[int, str], List[Tuple[str | Path, str]]] = defaultdict(
        list
    )
    for path in paths:
        exp_info = load_exp_info(path)

        # Use source_lang and target_lang for language identification
        source_lang = exp_info.get("source_lang", "unknown")
        target_lang = exp_info.get("target_lang", None)

        # Create language key distinguishing monolingual vs multilingual
        if target_lang is None:
            lang_key = source_lang  # Monolingual: e.g., "de"
        else:
            lang_key = f"{source_lang}_{target_lang}"  # Multilingual: e.g., "de_en"

        config_key = (exp_info.get("concat_size", 0), lang_key)
        config_groups[config_key].append((path, exp_info.get("model_name", "")))

    # Pre-compute ALL results for ALL models to build the dictionary
    all_model_results: Dict[Tuple[Tuple[int, str], str], Dict[str, Any]] = {}

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

            # Store result for later use (keep key structure identical to plotting code)
            result_key = (config_key, model_name)
            all_model_results[result_key] = result

    return all_model_results


def compute_position_diff_metrics(
    all_model_results: Dict[Tuple[Tuple[int, str], str], Dict[str, Any]],
) -> Dict[Tuple[Tuple[int, str], str], Dict[str, float]]:
    """
    Compute average differences, t-stats/p-values, and Wilcoxon signed-rank p-values between selected positions for each experiment setup.

    Definitions:
    - "1st" -> position index 0
    - "2nd" -> position index 1
    - "last" -> position index (concat_size - 1)
    - Average difference is defined as mean(posA) - mean(posB), using position_means
    - t-statistics/p-values are from a two-sided Welch's t-test (unequal variances) comparing
        the raw similarity samples of the respective positions
    - Wilcoxon signed-rank p-values use paired samples per document (two-sided)

    Args:
        all_model_results: Output from collect_multi_model_position_analysis_results.

    Returns:
        Dict keyed like the input, mapping to metrics:
        {
            key: {
                "avg_diff_pos1_pos2_abs": float,
                "avg_diff_pos1_posLast_abs": float,
                "avg_diff_pos2_posLast_abs": float,
                "avg_diff_pos1_pos2_rel": float,
                "avg_diff_pos1_posLast_rel": float,
                "avg_diff_pos2_posLast_rel": float,
                "avg_diff_pos1_pos2_rel_ci_low": float,
                "avg_diff_pos1_pos2_rel_ci_high": float,
                "avg_diff_pos1_posLast_rel_ci_low": float,
                "avg_diff_pos1_posLast_rel_ci_high": float,
                "avg_diff_pos2_posLast_rel_ci_low": float,
                "avg_diff_pos2_posLast_rel_ci_high": float,
                "t_stat_pos1_pos2": float,
                "t_stat_pos1_posLast": float,
                "t_stat_pos2_posLast": float,
                "p_value_pos1_pos2": float,
                "p_value_pos1_posLast": float,
                "p_value_pos2_posLast": float,
                "wilcoxon_stat_pos1_pos2": float,
                "wilcoxon_stat_pos1_posLast": float,
                "wilcoxon_stat_pos2_posLast": float,
                "wilcoxon_p_value_pos1_pos2": float,
                "wilcoxon_p_value_pos1_posLast": float,
                "wilcoxon_p_value_pos2_posLast": float,
            },
            ...
        }
    """
    metrics: Dict[Tuple[Tuple[int, str], str], Dict[str, float]] = {}

    def _rel_and_ci(x: List[float], y: List[float]) -> Tuple[float, float, float]:
        """Compute relative diff (mean(x) - mean(y)) / mean(x) and 95% CI via delta method."""
        assert len(x) == len(y) and len(x) >= 2, "Need paired samples with n>=2."
        mx = float(np.mean(x))
        my = float(np.mean(y))
        assert (
            mx != 0.0
        ), "Relative difference denominator (first position mean) is zero."
        n = len(x)
        vx = float(np.var(x, ddof=1))
        vy = float(np.var(y, ddof=1))
        cxy = float(np.cov(x, y, ddof=1)[0, 1])
        vmx = vx / n
        vmy = vy / n
        cmxy = cxy / n
        # f(mx,my) = 1 - my/mx
        dmx = my / (mx * mx)
        dmy = -1.0 / mx
        var_rel = (dmx * dmx) * vmx + (dmy * dmy) * vmy + 2.0 * dmx * dmy * cmxy
        if var_rel < 0.0:
            var_rel = 0.0
        se = float(np.sqrt(var_rel))
        tcrit = float(stats.t.ppf(0.975, n - 1))
        rel = float((mx - my) / mx)
        lo = rel - tcrit * se
        hi = rel + tcrit * se
        return rel, lo, hi

    for key, result in all_model_results.items():
        # Fail fast on missing keys
        assert (
            "position_similarities" in result
        ), "Missing 'position_similarities' in result."
        assert "position_means" in result, "Missing 'position_means' in result."
        assert "concat_size" in result, "Missing 'concat_size' in result."

        sims: List[List[float]] = result["position_similarities"]
        means: List[float] = result["position_means"]
        concat_size: int = int(result["concat_size"])  # type: ignore[arg-type]

        # Basic structural assumptions
        assert isinstance(sims, list) and all(isinstance(x, list) for x in sims)
        assert isinstance(means, list)
        assert len(sims) >= 2, "Need at least 2 positions to compare (first vs second)."
        assert len(means) == len(sims), "Lengths of means and similarities must match."
        assert concat_size == len(sims), "concat_size must equal number of positions."

        first_idx = 0
        second_idx = 1
        last_idx = len(sims) - 1

        # Non-empty groups for tests
        a = sims[first_idx]
        b = sims[second_idx]
        c = sims[last_idx]

        assert len(a) > 0 and len(b) > 0, "Positions 1 and 2 must have samples."
        assert len(c) > 0, "Last position must have samples."
        # For Wilcoxon we require paired samples (same length), which holds for document-level analysis
        assert (
            len(a) == len(b) == len(c)
        ), "Wilcoxon requires paired samples of equal length."

        # Average differences from means (absolute)
        avg_diff_pos1_pos2_abs = float(means[first_idx] - means[second_idx])
        avg_diff_pos1_posLast_abs = float(means[first_idx] - means[last_idx])
        avg_diff_pos2_posLast_abs = float(means[second_idx] - means[last_idx])

        # Relative differences and 95% CIs (denominator = first position mean)
        avg_diff_pos1_pos2_rel, rel12_lo, rel12_hi = _rel_and_ci(a, b)
        avg_diff_pos1_posLast_rel, rel1L_lo, rel1L_hi = _rel_and_ci(a, c)
        avg_diff_pos2_posLast_rel, rel2L_lo, rel2L_hi = _rel_and_ci(b, c)

        # Welch's t-tests (two-sided)
        t12 = stats.ttest_ind(a, b, equal_var=False)
        t1L = stats.ttest_ind(a, c, equal_var=False)
        t2L = stats.ttest_ind(b, c, equal_var=False)

        # Wilcoxon signed-rank tests (two-sided, paired)
        w12 = stats.wilcoxon(a, b, alternative="two-sided")
        w1L = stats.wilcoxon(a, c, alternative="two-sided")
        w2L = stats.wilcoxon(b, c, alternative="two-sided")

        # Prepare rounded outputs: absolute diffs (3 decimals), relative as percentages (1 decimal)
        abs12 = round(avg_diff_pos1_pos2_abs, 3)
        abs1L = round(avg_diff_pos1_posLast_abs, 3)
        abs2L = round(avg_diff_pos2_posLast_abs, 3)
        rel12_pct = round(avg_diff_pos1_pos2_rel * 100.0, 1)
        rel1L_pct = round(avg_diff_pos1_posLast_rel * 100.0, 1)
        rel2L_pct = round(avg_diff_pos2_posLast_rel * 100.0, 1)

        abs12_FULL = avg_diff_pos1_pos2_abs
        abs1L_FULL = avg_diff_pos1_posLast_abs
        abs2L_FULL = avg_diff_pos2_posLast_abs
        rel12_pct_FULL = avg_diff_pos1_pos2_rel * 100.0
        rel1L_pct_FULL = avg_diff_pos1_posLast_rel * 100.0
        rel2L_pct_FULL = avg_diff_pos2_posLast_rel * 100.0

        rel12_lo_pct = round(rel12_lo * 100.0, 1)
        rel12_hi_pct = round(rel12_hi * 100.0, 1)
        rel1L_lo_pct = round(rel1L_lo * 100.0, 1)
        rel1L_hi_pct = round(rel1L_hi * 100.0, 1)
        rel2L_lo_pct = round(rel2L_lo * 100.0, 1)
        rel2L_hi_pct = round(rel2L_hi * 100.0, 1)

        metrics[key] = {
            "avg_diff_pos1_pos2_abs": abs12,
            "avg_diff_pos1_posLast_abs": abs1L,
            "avg_diff_pos2_posLast_abs": abs2L,
            "avg_diff_pos1_pos2_rel": rel12_pct,
            "avg_diff_pos1_posLast_rel": rel1L_pct,
            "avg_diff_pos2_posLast_rel": rel2L_pct,
            "avg_diff_pos1_pos2_abs_FULL": abs12_FULL,
            "avg_diff_pos1_posLast_abs_FULL": abs1L_FULL,
            "avg_diff_pos2_posLast_abs_FULL": abs2L_FULL,
            "avg_diff_pos1_pos2_rel_FULL": rel12_pct_FULL,
            "avg_diff_pos1_posLast_rel_FULL": rel1L_pct_FULL,
            "avg_diff_pos2_posLast_rel_FULL": rel2L_pct_FULL,
            "avg_diff_pos1_pos2_rel_ci_low": rel12_lo_pct,
            "avg_diff_pos1_pos2_rel_ci_high": rel12_hi_pct,
            "avg_diff_pos1_posLast_rel_ci_low": rel1L_lo_pct,
            "avg_diff_pos1_posLast_rel_ci_high": rel1L_hi_pct,
            "avg_diff_pos2_posLast_rel_ci_low": rel2L_lo_pct,
            "avg_diff_pos2_posLast_rel_ci_high": rel2L_hi_pct,
            "t_stat_pos1_pos2": float(t12.statistic),
            "t_stat_pos1_posLast": float(t1L.statistic),
            "t_stat_pos2_posLast": float(t2L.statistic),
            "p_value_pos1_pos2": float(t12.pvalue),
            "p_value_pos1_posLast": float(t1L.pvalue),
            "p_value_pos2_posLast": float(t2L.pvalue),
            "wilcoxon_stat_pos1_pos2": float(w12.statistic),
            "wilcoxon_stat_pos1_posLast": float(w1L.statistic),
            "wilcoxon_stat_pos2_posLast": float(w2L.statistic),
            "wilcoxon_p_value_pos1_pos2": float(w12.pvalue),
            "wilcoxon_p_value_pos1_posLast": float(w1L.pvalue),
            "wilcoxon_p_value_pos2_posLast": float(w2L.pvalue),
        }

    return metrics


def compute_position_statistical_metrics(
    all_model_results: Dict[Tuple[Tuple[int, str], str], Dict[str, Any]],
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[Tuple[Tuple[int, str], str], Dict[str, Any]]:
    """Compute per-instance positional statistics: repeated-measures ANOVA, clustered OLS betas, Gini index with permutation p-value.

    Inputs:
        all_model_results: Output of collect_multi_model_position_analysis_results.
        n_permutations: Number of permutations for Gini p-value (one-sided, G > 0).
        seed: RNG seed for reproducibility.

    Returns:
        Dict keyed like input with metrics:
            {
              'anova_p_value': float,
              'anova_eta_squared': float,  # partial eta^2
              'ols_betas': {pos_index: float},
              'ols_p_values': {pos_index: float},
              'gini_index': float,
              'gini_p_value': float,
            }
    """
    if not all_model_results:
        return {}

    rng = np.random.default_rng(seed)
    out: Dict[Tuple[Tuple[int, str], str], Dict[str, Any]] = {}

    for key, result in all_model_results.items():
        # Fail fast on required keys
        assert (
            "position_similarities" in result
        ), "Missing 'position_similarities' in result."
        assert (
            "position_docID_segID" in result
        ), "Missing 'position_docID_segID' in result."
        sims_by_pos: List[List[float]] = result["position_similarities"]
        meta_by_pos: List[List[Tuple[str, str]]] = result["position_docID_segID"]

        # Structural asserts
        assert isinstance(sims_by_pos, list) and all(
            isinstance(x, list) for x in sims_by_pos
        )
        assert isinstance(meta_by_pos, list) and len(meta_by_pos) == len(sims_by_pos)
        P = len(sims_by_pos)
        assert P >= 2, "Need at least two positions."

        # Build long dataframe of raw (doc, item, position, similarity) rows
        records: List[Dict[str, Any]] = []
        for pos_idx, (vals, metas) in enumerate(zip(sims_by_pos, meta_by_pos)):
            assert len(vals) == len(metas), "Similarity/meta length mismatch."
            for (doc_id, seg_id), sim in zip(metas, vals):
                # Derive item id by sorting the segment IDs in doc_id
                # Assumption: doc_id segments separated by '_' and represent the set (order varies by permutation)
                item_id = "_".join(sorted(doc_id.split("_")))
                records.append(
                    {
                        "item": item_id,
                        "doc": doc_id,
                        "seg": seg_id,
                        "position": pos_idx,  # 0-based
                        "similarity": float(sim),
                    }
                )

        df_long = pd.DataFrame.from_records(records)
        assert not df_long.empty, "No data assembled."
        assert (
            df_long["similarity"].map(np.isfinite).all()
        ), "Non-finite similarity values."

        # Compute per-item per-position means (μ_{i,j})
        grouped = df_long.groupby(["item", "position"])["similarity"].mean()
        # Pivot to matrix X shape (M, P)
        df_wide = grouped.unstack("position")
        # Ensure all positions present for every item (balanced)
        assert (
            df_wide.shape[1] == P
        ), "Unbalanced positions (some items missing a position)."
        assert df_wide.notna().all(axis=None), "Missing values after pivot."
        X = df_wide.to_numpy()  # (M, P)
        M = X.shape[0]
        assert M >= 2, "Need at least two items for repeated-measures ANOVA."

        # Repeated-measures ANOVA using pingouin on per-item means
        df_itempos = grouped.reset_index()  # columns: item, position, similarity
        # pingouin expects within factor as categorical; keep ints
        aov = pg.rm_anova(
            data=df_itempos,
            dv="similarity",
            within="position",
            subject="item",
            detailed=True,
        )
        aov_row = aov[aov["Source"] == "position"]
        assert not aov_row.empty, "ANOVA result missing position row."
        p_value = float(aov_row["p-unc"].iloc[0])
        # Handle possible absence of 'ng2' depending on pingouin version
        if "ng2" in aov_row.columns:
            eta_sq_generalized = float(aov_row["ng2"].iloc[0])
        assert 0.0 <= eta_sq_generalized <= 1.0
        pos_means = X.mean(axis=0)

        # Clustered OLS with formula and categorical position including intercept
        df_long["position_cat"] = df_long["position"].astype("category")
        ols_model = smf.ols("similarity ~ C(position_cat)", data=df_long)
        ols_fit = ols_model.fit(
            cov_type="cluster", cov_kwds={"groups": df_long["item"]}
        )
        cov = ols_fit.cov_params()
        params = ols_fit.params
        df_resid = ols_fit.df_resid
        # Raw coefficient-level OLS summary (no adjusted per-position means)
        params_full = ols_fit.params
        se_full = ols_fit.bse
        conf_full = ols_fit.conf_int(alpha=0.05)
        pvals_full = ols_fit.pvalues
        ols_summary: Dict[str, Dict[str, float]] = {}
        for coef_name in params_full.index:
            beta = float(params_full[coef_name])
            se_coef = float(se_full[coef_name])
            ci_low, ci_high = map(float, conf_full.loc[coef_name])
            pval = float(pvals_full[coef_name])
            ols_summary[str(coef_name)] = {
                "beta": beta,
                "se": se_coef,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "pval": pval,
            }

        # Position 0
        means_per_pos: Dict[int, float] = {}
        pvals_per_pos: Dict[int, float] = {}
        intercept_name = "Intercept"
        means_per_pos[0] = float(params[intercept_name])
        var0 = float(cov.loc[intercept_name, intercept_name])
        assert var0 >= 0.0
        t0 = means_per_pos[0] / np.sqrt(var0) if var0 > 0 else 0.0
        pvals_per_pos[0] = 2 * float(stats.t.sf(abs(t0), df_resid)) if var0 > 0 else 1.0
        # Other positions
        for pos_idx in range(1, P):
            coef_name = f"C(position_cat)[T.{pos_idx}]"
            assert coef_name in params, f"Missing coefficient {coef_name}"
            mean_pos = float(params[intercept_name] + params[coef_name])
            var_pos = (
                float(cov.loc[intercept_name, intercept_name])
                + float(cov.loc[coef_name, coef_name])
                + 2 * float(cov.loc[intercept_name, coef_name])
            )
            if var_pos < 0 and abs(var_pos) < 1e-12:
                var_pos = 0.0
            assert var_pos >= 0.0
            t_stat = mean_pos / np.sqrt(var_pos) if var_pos > 0 else 0.0
            p_val = 2 * float(stats.t.sf(abs(t_stat), df_resid)) if var_pos > 0 else 1.0
            means_per_pos[pos_idx] = mean_pos
            pvals_per_pos[pos_idx] = p_val
        ols_betas_adjusted = means_per_pos
        ols_pvals_adjusted = pvals_per_pos
        ols_betas_raw = {str(k): float(v) for k, v in params.items()}
        ols_pvals_raw = {str(k): float(v) for k, v in ols_fit.pvalues.items()}

        # Gini index on per-position means with permutation p-value
        mean_overall = float(pos_means.mean())
        assert mean_overall != 0.0, "Overall mean is zero (Gini undefined)."
        diffs = np.abs(pos_means[:, None] - pos_means[None, :])
        gini = float(diffs.sum() / (2 * (P**2) * mean_overall))
        greater_equal = 0
        for _ in range(n_permutations):
            X_perm = np.array([rng.permutation(row) for row in X])
            pos_means_perm = X_perm.mean(axis=0)
            mean_perm = float(pos_means_perm.mean())
            if mean_perm == 0.0:
                continue
            diffs_perm = np.abs(pos_means_perm[:, None] - pos_means_perm[None, :])
            g_perm = diffs_perm.sum() / (2 * (P**2) * mean_perm)
            if g_perm >= gini:
                greater_equal += 1
        gini_p = (greater_equal + 1) / (n_permutations + 1)

        out[key] = {
            "anova_p_value": p_value,
            "anova_eta_squared_generalized": eta_sq_generalized,
            "ols_betas_adjusted": ols_betas_adjusted,
            "ols_p_values_adjusted": ols_pvals_adjusted,
            "ols_betas_raw": ols_betas_raw,
            "ols_p_values_raw": ols_pvals_raw,
            "ols_summary": ols_summary,
            "gini_index": gini,
            "gini_p_value": gini_p,
        }

    return out


def build_ols_coefficients_table_latex(
    stats_results: Dict[Tuple[Tuple[int, str], str], Dict[str, Any]],
    model_name: str,
    languages: List[str],
    ns: List[int],
) -> str:
    """Create a LaTeX table of raw OLS coefficients for a given model.

    The table includes Intercept and C(position_cat)[T.i] coefficients per n and language.
    """
    assert isinstance(stats_results, dict) and len(stats_results) > 0
    assert isinstance(model_name, str) and len(model_name) > 0
    assert (
        isinstance(languages, list)
        and len(languages) > 0
        and all(isinstance(l, str) and len(l) > 0 for l in languages)
    )
    assert (
        isinstance(ns, list)
        and len(ns) > 0
        and all(isinstance(n, int) and n >= 2 for n in ns)
    )

    def _stars(p: float) -> str:
        assert np.isfinite(p) and 0.0 <= p <= 1.0
        if p < 1e-3:
            return "***"
        if p < 1e-2:
            return "**"
        if p < 5e-2:
            return "*"
        return ""

    def _fmt(beta: float, p: float) -> str:
        assert np.isfinite(beta) and np.isfinite(p)
        return f"{beta:.2f}{_stars(p)}"

    # Header and table setup
    num_cols_total = 2 + len(languages)
    lines: List[str] = []
    lines.append("\\begin{table}[tb]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\setlength{\\tabcolsep}{6pt}")
    lines.append(
        f"  \\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}} ll *{{{len(languages)}}}{{c}} @{{}}}}"
    )
    lines.append("    \\toprule")
    lines.append(
        f"    \\multicolumn{{{num_cols_total}}}{{c}}{{\\textbf{{OLS Coefficients (raw), {model_name}}}}} \\\\"
    )
    lines.append("    \\addlinespace[2ex]")
    header_cols = " & ".join(languages)
    lines.append(f"    $n$& & {header_cols} \\\\")
    lines.append("    \\midrule")

    # Body: for each n block produce n rows (Intercept + T.1..T.(n-1))
    for idx_n, n in enumerate(ns):
        # Intercept row
        row_cells: List[str] = []
        for lang in languages:
            key = ((n, lang), model_name)
            assert (
                key in stats_results
            ), f"Missing results for (n={n}, lang={lang}, model={model_name})."
            res = stats_results[key]
            assert "ols_summary" in res, "Missing 'ols_summary' in stats_results entry."
            ols_sum = res["ols_summary"]
            assert "Intercept" in ols_sum, "Missing 'Intercept' coefficient."
            beta = float(ols_sum["Intercept"]["beta"])  # type: ignore[index]
            pval = float(ols_sum["Intercept"]["pval"])  # type: ignore[index]
            row_cells.append(_fmt(beta, pval))
        multirow_prefix = f"    \\multirow{{{n}}}{{*}}{{{n}}}"
        lines.append(
            f"{multirow_prefix} & $\\olsBetai{{0}}$ $(\\olsIntercept)$ & "
            + " & ".join(row_cells)
            + " \\\\"
        )

        # Subsequent rows for T.1..T.(n-1)
        for i in range(1, n):
            row_cells = []
            coef_name = f"C(position_cat)[T.{i}]"
            for lang in languages:
                key = ((n, lang), model_name)
                ols_sum = stats_results[key]["ols_summary"]
                assert (
                    coef_name in ols_sum
                ), f"Missing coefficient '{coef_name}' for (n={n}, lang={lang})."
                beta = float(ols_sum[coef_name]["beta"])  # type: ignore[index]
                pval = float(ols_sum[coef_name]["pval"])  # type: ignore[index]
                row_cells.append(_fmt(beta, pval))
            lines.append(
                f"    & $\\olsBetai{{{i+1}}}$ $(\\olsPi{{{i+1}}})$ & "
                + " & ".join(row_cells)
                + " \\\\"
            )

        if idx_n != len(ns) - 1:
            lines.append("    \\addlinespace[2ex]")

    # Footer
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular*}")
    lines.append(
        "  \\caption{OLS coefficients (raw). Stars indicate significance: *$p<0.05$, **$p<0.01$, ***$p<0.001$.}"
    )
    # Label derived from model_name (sanitized)
    label_safe = (
        model_name.lower().replace("/", "-").replace(" ", "-").replace("_", "-")
    )
    lines.append(f"  \\label{{tab:ols-coef-{label_safe}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def build_gini_index_table_latex(
    stats_results: Dict[Tuple[Tuple[int, str], str], Dict[str, Any]],
    model_name: str,
    languages: List[str],
    ns: List[int],
) -> str:
    """Create a LaTeX table of Gini index scores for a given model.

    The table has one row per n (concat size) and one column per language.
    Values are the rounded Gini indices with significance stars from gini_p_value.
    """
    # Basic assertions (fail fast)
    assert isinstance(stats_results, dict) and len(stats_results) > 0
    assert isinstance(model_name, str) and len(model_name) > 0
    assert (
        isinstance(languages, list)
        and len(languages) > 0
        and all(isinstance(l, str) and len(l) > 0 for l in languages)
    )
    assert (
        isinstance(ns, list)
        and len(ns) > 0
        and all(isinstance(n, int) and n >= 2 for n in ns)
    )

    def _stars(p: float) -> str:
        assert np.isfinite(p) and 0.0 <= p <= 1.0
        if p < 1e-3:
            return "***"
        if p < 1e-2:
            return "**"
        if p < 5e-2:
            return "*"
        return ""

    def _fmt(val: float, p: float) -> str:
        assert np.isfinite(val) and np.isfinite(p)
        return f"{val:.2f}{_stars(p)}"

    num_cols_total = 1 + len(languages)
    lines: List[str] = []
    lines.append("\\begin{table}[tb]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\setlength{\\tabcolsep}{6pt}")
    lines.append(
        f"  \\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}} l *{{{len(languages)}}}{{c}} @{{}}}}"
    )
    lines.append("    \\toprule")
    lines.append(
        f"    \\multicolumn{{{num_cols_total}}}{{c}}{{\\textbf{{GINI Index scores, {model_name}}}}} \\\\"
    )
    lines.append("    \\addlinespace[2ex]")
    header_cols = " & ".join(languages)
    lines.append(f"    $n$ & {header_cols} \\\\")
    lines.append("    \\midrule")

    for idx_n, n in enumerate(ns):
        row_cells: List[str] = []
        for lang in languages:
            key = ((n, lang), model_name)
            assert (
                key in stats_results
            ), f"Missing results for (n={n}, lang={lang}, model={model_name})."
            res = stats_results[key]
            assert "gini_index" in res and "gini_p_value" in res
            g = float(res["gini_index"])  # type: ignore[index]
            p = float(res["gini_p_value"])  # type: ignore[index]
            # Gini is defined in [0,1]
            assert 0.0 <= g <= 1.0 and np.isfinite(g)
            row_cells.append(_fmt(g, p))
        lines.append(f"    {n} & " + " & ".join(row_cells) + " \\\\")
        if idx_n != len(ns) - 1:
            lines.append("    \\addlinespace[2ex]")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular*}")
    lines.append(
        "  \\caption{GINI Index scores (scores are within [0, 1]). Stars indicate significance: *$p<0.05$, **$p<0.01$, ***$p<0.001$.}"
    )
    label_safe = (
        model_name.lower().replace("/", "-").replace(" ", "-").replace("_", "-")
    )
    lines.append(f"  \\label{{tab:gini-{label_safe}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def build_anova_results_table_latex(
    stats_results: Dict[Tuple[Tuple[int, str], str], Dict[str, Any]],
    model_name: str,
    languages: List[str],
    ns: List[int],
) -> str:
    """Create a LaTeX table (tabularx) of ANOVA results for a given model.

    The table has two rows per n (concat size): p-value and generalized eta squared.
    Values are rounded as follows:
      - anova_p_value: 3 decimals
      - anova_eta_squared_generalized: 2 decimals
    """
    # Basic assertions (fail fast)
    assert isinstance(stats_results, dict) and len(stats_results) > 0
    assert isinstance(model_name, str) and len(model_name) > 0
    assert (
        isinstance(languages, list)
        and len(languages) > 0
        and all(isinstance(l, str) and len(l) > 0 for l in languages)
    )
    assert (
        isinstance(ns, list)
        and len(ns) > 0
        and all(isinstance(n, int) and n >= 2 for n in ns)
    )

    def _fmt_p(p: float) -> str:
        assert np.isfinite(p) and 0.0 <= p <= 1.0
        return f"{p:.3f}"

    def _fmt_eta(e: float) -> str:
        assert np.isfinite(e) and 0.0 <= e <= 1.0
        return f"{e:.2f}"

    num_cols_total = 2 + len(languages)
    lines: List[str] = []
    lines.append("\\begin{table}[tb]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\setlength{\\tabcolsep}{6pt}")
    lines.append(
        f"  \\begin{{tabularx}}{{\\linewidth}}{{*{{2}}{{>{{\\raggedright\\arraybackslash}}X}} *{{{len(languages)}}}{{>{{\\centering\\arraybackslash}}X}}}}"
    )
    lines.append("    \\toprule")
    lines.append(
        f"    \\multicolumn{{{num_cols_total}}}{{c}}{{\\textbf{{Anova Results, {model_name}}}}} \\\\"
    )
    lines.append("    \\addlinespace[2ex]")
    header_cols = " & ".join(languages)
    lines.append(f"    $n$& & {header_cols} \\\\")
    lines.append("    \\midrule")

    for idx_n, n in enumerate(ns):
        # p-values row
        pvals: List[str] = []
        etas: List[str] = []
        for lang in languages:
            key = ((n, lang), model_name)
            assert (
                key in stats_results
            ), f"Missing results for (n={n}, lang={lang}, model={model_name})."
            res = stats_results[key]
            assert (
                "anova_p_value" in res and "anova_eta_squared_generalized" in res
            ), "Missing ANOVA keys in stats_results entry."
            p = float(res["anova_p_value"])  # type: ignore[index]
            e = float(res["anova_eta_squared_generalized"])  # type: ignore[index]
            pvals.append(_fmt_p(p))
            etas.append(_fmt_eta(e))

        lines.append(
            f"    \\multirow{{2}}{{*}}{{{n}}} & $\\pVal$ & "
            + " & ".join(pvals)
            + " \\\\"
        )
        lines.append("    & $\\etaSquaredG$ & " + " & ".join(etas) + " \\\\")
        if idx_n != len(ns) - 1:
            lines.append("    \\addlinespace[2ex]")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabularx}")
    lines.append("  \\caption{Anova results.}")
    label_safe = (
        model_name.lower().replace("/", "-").replace(" ", "-").replace("_", "-")
    )
    lines.append(f"  \\label{{tab:anova-{label_safe}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def build_dataset_sample_table(dataset: Any, dataset_id: int, length: int) -> str:
    """Create a LaTeX table showing truncated text samples per language split."""
    assert dataset is not None, "dataset must be provided"
    assert isinstance(dataset_id, int) and dataset_id >= 0, "dataset_id must be >= 0"
    assert isinstance(length, int) and length > 0, "length must be > 0"

    lang_rows: List[Tuple[str, str, str]] = [
        ("en", "English", ""),
        ("zh", "Chinese", "\\CJKfont "),
        ("de", "German", ""),
        ("it", "Italian", ""),
        ("ko", "Korean", "\\KRfont "),
        ("hi", "Hindi", "\\DEVfont "),
    ]

    def _escape_latex(text: str) -> str:
        replacements = {
            "\\": r"\\textbackslash{}",
            "&": r"\\&",
            "%": r"\\%",
            "$": r"\\$",
            "#": r"\\#",
            "_": r"\\_",
            "{": r"\\{",
            "}": r"\\}",
            "~": r"\\textasciitilde{}",
            "^": r"\\textasciicircum{}",
        }
        escaped = text
        for key, val in replacements.items():
            escaped = escaped.replace(key, val)
        return escaped

    rows: List[str] = []
    for idx, (lang_code, display_name, macro) in enumerate(lang_rows):
        assert lang_code in dataset, f"Missing language split '{lang_code}'"
        split = dataset[lang_code]
        assert dataset_id < len(
            split
        ), f"dataset_id {dataset_id} out of range for '{lang_code}'"
        record = split[dataset_id]
        assert "text" in record, "Missing 'text' field in dataset record"
        text = record["text"]
        assert isinstance(text, str), "Record 'text' must be a string"
        snippet_raw = text[:length]
        snippet_compact = " ".join(snippet_raw.split())
        snippet = _escape_latex(snippet_compact)
        content = f"{macro}{snippet}" if macro else snippet
        rows.append(f"    \\textbf{{{display_name}}} & {{{content}}} \\\\")
        if idx != len(lang_rows) - 1:
            rows.append("    \\addlinespace[2ex]")

    lines: List[str] = []
    lines.append("\\begin{table}[tb]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\setlength{\\tabcolsep}{6pt}")
    lines.append(
        "  \\begin{tabularx}{\\linewidth}{@{}l >{\\RaggedRight\\arraybackslash}X@{}}"
    )
    lines.append("    \\toprule")
    lines.append("    \\addlinespace[2ex]")
    lines.extend(rows)
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabularx}")
    lines.append("  \\label{tab:wiki-comparable-sample}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def categorize_paths_by_root(
    parallel_dir: str, relative_root: str, language_order: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """
    Categorize paths based on a given root into monolingual and multilingual lists.

    Args:
        relative_root (str): The root pattern up to and including "parallel__"
                           e.g., "jinaai_jina-embeddings-v3__wiki_parallel_en_de_hi_it_ko_zh__parallel__"
        language_order (Optional[List[str]]): Custom order for language codes.
                                             e.g., ["en", "de", "it", "ko", "hi"]
                                             If None, paths are sorted alphabetically.

    Returns:
        Tuple[List[str], List[str]]: (monolingual_paths, multilingual_paths)

    Example:
        >>> mono, multi = categorize_paths_by_root(
        ...     "jinaai_jina-embeddings-v3__wiki_parallel_en_de_hi_it_ko__parallel__",
        ...     language_order=["en", "de", "it", "ko", "hi"]
        ... )
        >>> print(f"Found {len(mono)} monolingual and {len(multi)} multilingual paths")
    """
    import os
    import glob
    from typing import Tuple, List, Optional
    from collections import defaultdict
    import itertools

    pattern = os.path.join(parallel_dir, "*")
    all_subdirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]

    root_pattern = relative_root.rstrip("/\\")

    # Group paths by language combination and then by concat size
    mono_groups = defaultdict(list)  # {language: [paths]}
    multi_groups = defaultdict(list)  # {language_combination: [paths]}

    for subdir in all_subdirs:
        dir_name = os.path.basename(subdir)

        if dir_name.startswith(root_pattern):
            remaining = dir_name[len(root_pattern) :]

            if remaining and "__concat-size" in remaining:
                parts = remaining.split("__")
                if len(parts) >= 2:
                    language_part = parts[0]

                    if "_" in language_part:
                        # Multilingual: contains underscore (e.g., "en_de", "hi_it")
                        multi_groups[language_part].append(os.path.relpath(subdir))
                    else:
                        # Monolingual: single language code (e.g., "en", "de", "hi")
                        mono_groups[language_part].append(os.path.relpath(subdir))

    # Sort each group by the rest of the path (concat-size, range, etc.)
    for lang in mono_groups:
        mono_groups[lang].sort()
    for lang_combo in multi_groups:
        multi_groups[lang_combo].sort()

    # Determine the final ordering
    if language_order is None:
        # Default: alphabetical sorting
        monolingual_paths = []
        for lang in sorted(mono_groups.keys()):
            monolingual_paths.extend(mono_groups[lang])

        multilingual_paths = []
        for lang_combo in sorted(multi_groups.keys()):
            multilingual_paths.extend(multi_groups[lang_combo])
    else:
        # Custom ordering based on language_order
        monolingual_paths = []

        # For monolingual: follow the exact order specified
        for lang in language_order:
            if lang in mono_groups:
                monolingual_paths.extend(mono_groups[lang])

        # Add any languages not in the specified order (in alphabetical order)
        remaining_mono_langs = set(mono_groups.keys()) - set(language_order)
        for lang in sorted(remaining_mono_langs):
            monolingual_paths.extend(mono_groups[lang])

        # For multilingual: create all combinations following the language_order
        multilingual_paths = []

        # Generate all possible pairs in the order specified
        lang_pairs_in_order = []
        for i, lang1 in enumerate(language_order):
            for j, lang2 in enumerate(language_order):
                if i != j:  # Different languages
                    # Both directions: lang1_lang2 and lang2_lang1
                    lang_pairs_in_order.append(f"{lang1}_{lang2}")

        # Add paths for each language combination in the specified order
        for lang_combo in lang_pairs_in_order:
            if lang_combo in multi_groups:
                multilingual_paths.extend(multi_groups[lang_combo])

        # Add any remaining combinations not covered by the language_order
        remaining_multi_combos = set(multi_groups.keys()) - set(lang_pairs_in_order)
        for lang_combo in sorted(remaining_multi_combos):
            multilingual_paths.extend(multi_groups[lang_combo])

    return monolingual_paths, multilingual_paths
