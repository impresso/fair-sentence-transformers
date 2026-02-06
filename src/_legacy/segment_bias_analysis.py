"""
Segment Bias Analysis Module for fair_sentence_transformers

This module provides tools to analyze potential biases in how embedding models
represent text segments within longer documents.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from scipy import stats
from collections import defaultdict


def compute_similarity_matrix(
    standalone_embeddings: Dict[str, np.ndarray],
    latechunk_embeddings: Dict[Tuple[str, int], np.ndarray],
) -> Tuple[np.ndarray, List[str], List[Tuple[str, int]]]:
    """
    Compute cosine similarity matrix between standalone and latechunk embeddings.

    Args:
        standalone_embeddings: Dict mapping segment IDs to their standalone embeddings
        latechunk_embeddings: Dict mapping (doc_id, segment_pos) to segment embeddings within docs

    Returns:
        similarity_matrix: Cosine similarity matrix
        standalone_ids: List of standalone segment IDs in order
        latechunk_ids: List of (doc_id, segment_pos) tuples in order
    """
    standalone_ids = list(standalone_embeddings.keys())
    latechunk_ids = list(latechunk_embeddings.keys())

    # Convert to matrices for efficient computation
    standalone_matrix = np.vstack(
        [standalone_embeddings[s_id] for s_id in standalone_ids]
    )
    latechunk_matrix = np.vstack(
        [latechunk_embeddings[lc_id] for lc_id in latechunk_ids]
    )

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(standalone_matrix, latechunk_matrix)

    return similarity_matrix, standalone_ids, latechunk_ids


def get_position_based_similarities(
    latechunk_embeddings: Dict[Tuple[str, int], np.ndarray],
    segment_map: Dict[Tuple[str, int], str],
    standalone_embeddings: Dict[str, np.ndarray],
) -> Dict[int, List[float]]:
    """
    Group cosine similarities by segment position.

    Args:
        latechunk_embeddings: Dict mapping (doc_id, segment_pos) to segment embeddings
        segment_map: Dict mapping (doc_id, segment_pos) to original segment ID
        standalone_embeddings: Dict mapping segment IDs to standalone embeddings

    Returns:
        pos_similarities: Dict mapping positions to lists of similarity values
    """
    position_similarities = defaultdict(list)

    for (doc_id, pos), latechunk_emb in latechunk_embeddings.items():
        segment_id = segment_map[(doc_id, pos)]
        standalone_emb = standalone_embeddings[segment_id]

        # Reshape to handle 1D arrays if needed
        latechunk_emb_reshaped = latechunk_emb.reshape(1, -1)
        standalone_emb_reshaped = standalone_emb.reshape(1, -1)

        # Calculate cosine similarity
        sim = cosine_similarity(latechunk_emb_reshaped, standalone_emb_reshaped)[0][0]
        position_similarities[pos].append(sim)

    return dict(position_similarities)


def analyze_cross_position_influence(
    latechunk_embeddings: Dict[Tuple[str, int], np.ndarray],
    segment_map: Dict[Tuple[str, int], str],
    standalone_embeddings: Dict[str, np.ndarray],
    positions: List[int],
) -> pd.DataFrame:
    """
    Analyze how much each position influences other positions.

    Args:
        latechunk_embeddings: Dict mapping (doc_id, segment_pos) to segment embeddings
        segment_map: Dict mapping (doc_id, segment_pos) to original segment ID
        standalone_embeddings: Dict mapping segment IDs to standalone embeddings
        positions: List of positions to analyze

    Returns:
        influence_df: DataFrame with cross-position influence metrics
    """
    # Create empty dataframe for position-to-position influence
    influence_df = pd.DataFrame(index=positions, columns=positions, dtype=float)

    # Group embeddings by document and position
    doc_pos_embeddings = defaultdict(dict)
    for (doc_id, pos), embedding in latechunk_embeddings.items():
        doc_pos_embeddings[doc_id][pos] = embedding

    # For each document
    for doc_id, pos_embeddings in doc_pos_embeddings.items():
        # Skip documents that don't have all the positions we want to analyze
        if not all(pos in pos_embeddings for pos in positions):
            continue

        # For each pair of positions
        for pos1 in positions:
            for pos2 in positions:
                segment_id1 = segment_map.get((doc_id, pos1))

                # Get embeddings
                latechunk_emb = pos_embeddings[pos1]
                standalone_emb = standalone_embeddings[segment_id1]
                other_pos_emb = pos_embeddings[pos2]

                # Calculate similarity between latechunk and standalone
                sim_to_standalone = cosine_similarity(
                    latechunk_emb.reshape(1, -1), standalone_emb.reshape(1, -1)
                )[0][0]

                # Calculate similarity between latechunk and other position
                sim_to_other_pos = cosine_similarity(
                    latechunk_emb.reshape(1, -1), other_pos_emb.reshape(1, -1)
                )[0][0]

                # Accumulate values (will average later)
                if pd.isna(influence_df.loc[pos1, pos2]):
                    influence_df.loc[pos1, pos2] = 0

                # Higher value means more influence
                influence_df.loc[pos1, pos2] += sim_to_other_pos / len(
                    doc_pos_embeddings
                )

    return influence_df


def visualize_embeddings_by_position(
    latechunk_embeddings: Dict[Tuple[str, int], np.ndarray],
    standalone_embeddings: Dict[str, np.ndarray] = None,
    segment_map: Dict[Tuple[str, int], str] = None,
    method: str = "tsne",
    n_components: int = 2,
) -> plt.Figure:
    """
    Visualize latechunk embeddings by their position in the document.
    Optionally include standalone embeddings for comparison.

    Args:
        latechunk_embeddings: Dict mapping (doc_id, segment_pos) to segment embeddings
        standalone_embeddings: Dict mapping segment IDs to standalone embeddings
        segment_map: Dict mapping (doc_id, segment_pos) to original segment ID
        method: Dimensionality reduction method ('tsne' or 'pca')
        n_components: Number of dimensions for visualization

    Returns:
        fig: Matplotlib figure with the visualization
    """
    # Extract positions and embeddings
    positions = []
    embeddings = []
    types = []
    segment_ids = []

    for (doc_id, pos), embedding in latechunk_embeddings.items():
        positions.append(pos)
        embeddings.append(embedding)
        types.append("latechunk")
        if segment_map:
            segment_ids.append(segment_map.get((doc_id, pos), "unknown"))
        else:
            segment_ids.append("unknown")

    # Include standalone embeddings if provided
    if standalone_embeddings and segment_map:
        for segment_id, embedding in standalone_embeddings.items():
            # Only include segments that appear in the latechunk data
            if segment_id in segment_ids:
                positions.append(-1)  # Special position for standalone
                embeddings.append(embedding)
                types.append("standalone")
                segment_ids.append(segment_id)

    # Convert to array
    embeddings_array = np.vstack(embeddings)

    # Apply dimensionality reduction
    if method.lower() == "tsne":
        reducer = TSNE(n_components=n_components, random_state=42)
    else:
        reducer = PCA(n_components=n_components)

    reduced_embeddings = reducer.fit_transform(embeddings_array)

    # Create DataFrame for plotting
    df = pd.DataFrame(
        {
            "x": reduced_embeddings[:, 0],
            "y": reduced_embeddings[:, 1],
            "position": [f"pos_{p}" if p >= 0 else "standalone" for p in positions],
            "type": types,
            "segment_id": segment_ids,
        }
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=df, x="x", y="y", hue="position", style="type", s=100, alpha=0.7, ax=ax
    )

    plt.title(f"Embedding Visualization using {method.upper()}")

    return fig


def analyze_cls_representation_bias(
    latechunk_cls_embeddings: Dict[str, np.ndarray],
    doc_segments: Dict[str, List[Tuple[str, int]]],
    segment_map: Dict[Tuple[str, int], str],
    standalone_embeddings: Dict[str, np.ndarray],
) -> Dict[int, float]:
    """
    Analyze how much each segment position contributes to the CLS representation.

    Args:
        latechunk_cls_embeddings: Dict mapping doc_id to CLS embeddings
        doc_segments: Dict mapping doc_id to list of segment positions
        segment_map: Dict mapping (doc_id, segment_pos) to original segment ID
        standalone_embeddings: Dict mapping segment IDs to standalone embeddings

    Returns:
        position_influence: Dict mapping positions to average influence scores
    """
    position_influence = defaultdict(list)

    # For each document
    for doc_id, cls_embedding in latechunk_cls_embeddings.items():
        # Get segments in this document
        segments = doc_segments.get(doc_id, [])

        # For each segment position
        for pos in sorted(set([pos for _, pos in segments])):
            # Get segment ID
            matching_segments = [
                (doc, p) for doc, p in segments if p == pos and doc == doc_id
            ]
            if not matching_segments:
                continue

            segment_id = segment_map.get(matching_segments[0])
            if not segment_id:
                continue

            # Get standalone embedding for this segment
            standalone_emb = standalone_embeddings.get(segment_id)
            if standalone_emb is None:
                continue

            # Calculate similarity between CLS and this segment's standalone embedding
            sim = cosine_similarity(
                cls_embedding.reshape(1, -1), standalone_emb.reshape(1, -1)
            )[0][0]

            # Record similarity by position
            position_influence[pos].append(sim)

    # Average the influences
    return {pos: np.mean(sims) for pos, sims in position_influence.items()}


def compare_segment_order_effect(
    latechunk_embeddings: Dict[Tuple[str, int], np.ndarray],
    reference_embeddings: Dict[str, np.ndarray],
    segment_map: Dict[Tuple[str, int], str],
    positions: List[int],
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[int, float]]]:
    """
    Compare the effect of segment order on embeddings.

    Args:
        latechunk_embeddings: Dict mapping (doc_id, segment_pos) to segment embeddings
        reference_embeddings: Dict mapping segment IDs to reference embeddings (standalone)
        segment_map: Dict mapping (doc_id, segment_pos) to original segment ID
        positions: List of positions to analyze

    Returns:
        position_metrics: Dict of position to metrics (correlation, distance)
        position_correlations: Dict of position pairs to correlation values
    """
    position_metrics = {pos: {"correlation": [], "distance": []} for pos in positions}
    position_correlations = {p1: {p2: [] for p2 in positions} for p1 in positions}

    # Group by document
    doc_positions = defaultdict(list)
    for doc_id, pos in latechunk_embeddings.keys():
        doc_positions[doc_id].append(pos)

    # For each document with multiple positions
    for doc_id, doc_pos_list in doc_positions.items():
        if len(doc_pos_list) < 2:
            continue

        # Get embeddings for each position
        pos_embeddings = {}
        for pos in doc_pos_list:
            if pos in positions:
                latechunk_emb = latechunk_embeddings.get((doc_id, pos))
                segment_id = segment_map.get((doc_id, pos))
                reference_emb = reference_embeddings.get(segment_id)

                if latechunk_emb is not None and reference_emb is not None:
                    pos_embeddings[pos] = {
                        "latechunk": latechunk_emb,
                        "reference": reference_emb,
                    }

        # Calculate metrics for each position
        for pos, embs in pos_embeddings.items():
            correlation = np.corrcoef(embs["latechunk"], embs["reference"])[0, 1]
            distance = np.linalg.norm(embs["latechunk"] - embs["reference"])

            position_metrics[pos]["correlation"].append(correlation)
            position_metrics[pos]["distance"].append(distance)

        # Calculate correlations between positions
        for pos1 in pos_embeddings:
            for pos2 in pos_embeddings:
                if pos1 != pos2:
                    corr = np.corrcoef(
                        pos_embeddings[pos1]["latechunk"],
                        pos_embeddings[pos2]["latechunk"],
                    )[0, 1]
                    position_correlations[pos1][pos2].append(corr)

    # Average the metrics
    for pos in position_metrics:
        for metric in position_metrics[pos]:
            if position_metrics[pos][metric]:
                position_metrics[pos][metric] = np.mean(position_metrics[pos][metric])
            else:
                position_metrics[pos][metric] = np.nan

    # Average the correlations
    for pos1 in position_correlations:
        for pos2 in position_correlations[pos1]:
            if position_correlations[pos1][pos2]:
                position_correlations[pos1][pos2] = np.mean(
                    position_correlations[pos1][pos2]
                )
            else:
                position_correlations[pos1][pos2] = np.nan

    return position_metrics, position_correlations


def information_leakage_analysis(
    latechunk_embeddings: Dict[Tuple[str, int], np.ndarray],
    segment_map: Dict[Tuple[str, int], str],
    standalone_embeddings: Dict[str, np.ndarray],
    positions: List[int],
) -> Dict[int, Dict[str, float]]:
    """
    Analyze information leakage between segments in different positions.

    Args:
        latechunk_embeddings: Dict mapping (doc_id, segment_pos) to segment embeddings
        segment_map: Dict mapping (doc_id, segment_pos) to original segment ID
        standalone_embeddings: Dict mapping segment IDs to standalone embeddings
        positions: List of positions to analyze

    Returns:
        leakage_metrics: Dictionary of positions and their leakage metrics
    """
    leakage_metrics = {
        pos: {"self_similarity": [], "other_similarity": []} for pos in positions
    }

    # Group by document
    doc_positions = defaultdict(dict)
    for (doc_id, pos), emb in latechunk_embeddings.items():
        if pos in positions:
            doc_positions[doc_id][pos] = {
                "latechunk": emb,
                "segment_id": segment_map.get((doc_id, pos)),
            }

    # For each document with multiple positions
    for doc_id, pos_data in doc_positions.items():
        if len(pos_data) < 2:
            continue

        for pos1 in pos_data:
            # Skip if missing data
            if pos1 not in positions or pos_data[pos1]["segment_id"] is None:
                continue

            # Get standalone embedding for this segment
            segment_id = pos_data[pos1]["segment_id"]
            standalone_emb = standalone_embeddings.get(segment_id)

            if standalone_emb is None:
                continue

            # Self similarity (latechunk embedding to its own standalone version)
            self_sim = cosine_similarity(
                pos_data[pos1]["latechunk"].reshape(1, -1),
                standalone_emb.reshape(1, -1),
            )[0][0]
            leakage_metrics[pos1]["self_similarity"].append(self_sim)

            # Similarity to other positions in same document
            other_sims = []
            for pos2 in pos_data:
                if pos1 != pos2 and pos2 in positions:
                    # Get standalone embedding for the other position
                    other_segment_id = pos_data[pos2]["segment_id"]
                    other_standalone_emb = standalone_embeddings.get(other_segment_id)

                    if other_standalone_emb is not None:
                        # Similarity between pos1's latechunk and pos2's standalone
                        other_sim = cosine_similarity(
                            pos_data[pos1]["latechunk"].reshape(1, -1),
                            other_standalone_emb.reshape(1, -1),
                        )[0][0]
                        other_sims.append(other_sim)

            if other_sims:
                # Average similarity to other positions
                leakage_metrics[pos1]["other_similarity"].append(np.mean(other_sims))

    # Average the metrics
    for pos in leakage_metrics:
        for metric in leakage_metrics[pos]:
            if leakage_metrics[pos][metric]:
                leakage_metrics[pos][metric] = np.mean(leakage_metrics[pos][metric])
            else:
                leakage_metrics[pos][metric] = np.nan

    return leakage_metrics
