"""Evaluation metrics for F1 race predictions."""

import numpy as np
import pandas as pd
from typing import List, Union, Dict
from scipy.stats import spearmanr, kendalltau


def spearman_rank_correlation(
    true_ranks: Union[List[int], np.ndarray, pd.Series],
    pred_ranks: Union[List[int], np.ndarray, pd.Series]
) -> float:
    """
    Calculate Spearman's rank correlation coefficient between true and predicted ranks.
    
    Args:
        true_ranks: Ground truth rankings.
        pred_ranks: Predicted rankings.
        
    Returns:
        Spearman's rank correlation coefficient.
    """
    return spearmanr(true_ranks, pred_ranks)[0]


def kendall_tau(
    true_ranks: Union[List[int], np.ndarray, pd.Series],
    pred_ranks: Union[List[int], np.ndarray, pd.Series]
) -> float:
    """
    Calculate Kendall's Tau between true and predicted ranks.
    
    Args:
        true_ranks: Ground truth rankings.
        pred_ranks: Predicted rankings.
        
    Returns:
        Kendall's Tau coefficient.
    """
    return kendalltau(true_ranks, pred_ranks)[0]


def ndcg_at_k(
    true_ranks: Union[List[int], np.ndarray, pd.Series],
    pred_ranks: Union[List[int], np.ndarray, pd.Series],
    k: int = 10
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.
    
    Args:
        true_ranks: Ground truth rankings (lower is better, e.g., 1 for 1st place).
        pred_ranks: Predicted rankings (lower is better, e.g., 1 for 1st place).
        k: Number of top positions to consider.
        
    Returns:
        NDCG@k score.
    """
    # Convert ranks to relevance scores (higher is better)
    # Assuming max rank possible is the length of the array
    max_rank = max(len(true_ranks), np.max(true_ranks) if len(true_ranks) > 0 else 0)
    true_relevance = max_rank + 1 - np.array(true_ranks)
    
    # Convert predicted ranks to ordering
    pred_ordering = np.argsort(np.array(pred_ranks))
    
    # Get true relevance in predicted order
    pred_relevance = true_relevance[pred_ordering]
    
    # Trim to top k
    pred_relevance = pred_relevance[:k]
    
    # Calculate DCG (Discounted Cumulative Gain)
    dcg = np.sum(pred_relevance / np.log2(np.arange(2, len(pred_relevance) + 2)))
    
    # Calculate ideal DCG
    ideal_ordering = np.argsort(-true_relevance)  # Sort by descending relevance
    ideal_relevance = true_relevance[ideal_ordering][:k]
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))
    
    if idcg == 0:
        return 0.0
        
    return dcg / idcg


def rmse_position(
    true_positions: Union[List[int], np.ndarray, pd.Series],
    pred_positions: Union[List[int], np.ndarray, pd.Series]
) -> float:
    """
    Calculate Root Mean Square Error of positions.
    
    Args:
        true_positions: Ground truth positions.
        pred_positions: Predicted positions.
        
    Returns:
        RMSE of positions.
    """
    return np.sqrt(np.mean((np.array(true_positions) - np.array(pred_positions)) ** 2))


def evaluate_all_metrics(
    true_ranks: Union[List[int], np.ndarray, pd.Series],
    pred_ranks: Union[List[int], np.ndarray, pd.Series],
    metrics: List[str] = None,
    ndcg_k: int = 10
) -> Dict[str, float]:
    """
    Calculate multiple evaluation metrics between true and predicted ranks.
    
    Args:
        true_ranks: Ground truth rankings.
        pred_ranks: Predicted rankings.
        metrics: List of metrics to calculate. Options are "spearman_rank_correlation",
                "kendall_tau", "ndcg_at_k", "rmse_position".
        ndcg_k: k value for NDCG@k metric.
        
    Returns:
        Dictionary of metric names and their values.
    """
    if metrics is None:
        metrics = ["spearman_rank_correlation", "kendall_tau", "ndcg_at_k", "rmse_position"]
    
    results = {}
    
    for metric in metrics:
        if metric == "spearman_rank_correlation":
            results[metric] = spearman_rank_correlation(true_ranks, pred_ranks)
        elif metric == "kendall_tau":
            results[metric] = kendall_tau(true_ranks, pred_ranks)
        elif metric == "ndcg_at_k":
            results[metric] = ndcg_at_k(true_ranks, pred_ranks, k=ndcg_k)
        elif metric == "rmse_position":
            results[metric] = rmse_position(true_ranks, pred_ranks)
    
    return results 