# =========================
# Utility Functions
# =========================

import os
import ast
import numpy as np
import pandas as pd
from typing import List, Tuple
from config import LEVEL_MAP, SURFACE_MAP, ROUND_MAP, LEAK_RE


def load_years(path: str, years: List[int], sep: str = ";") -> pd.DataFrame:
    """Load and concatenate CSV files for specified years."""
    dfs = []
    for y in years:
        fp = os.path.join(path, f"matches_with_odds_{y}.csv")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing file: {fp}")
        dfs.append(pd.read_csv(fp, sep=sep))
    return pd.concat(dfs, ignore_index=True)


def safe_to_datetime(s):
    """Safely convert to datetime with error handling."""
    return pd.to_datetime(s, errors="coerce")


def recent_win_rate(s, k: int = 10) -> float:
    """
    Calculate win rate from recent match history string.
    
    Args:
        s: String representation of match results list ['V','D',...]
        k: Number of recent matches to consider
        
    Returns:
        Win rate (0-1) or np.nan if invalid
    """
    try:
        lst = ast.literal_eval(s) if isinstance(s, str) else s
        if not isinstance(lst, list) or len(lst) == 0:
            return np.nan
        lst = lst[-k:]
        return sum(1 for x in lst if x == "V") / len(lst)
    except:
        return np.nan


def infer_player1_win_mapping(df: pd.DataFrame) -> int:
    """
    Auto-infer label mapping (Winner==1 or 0 means Player1 wins?).
    
    Heuristic: if player1 has better rank (smaller number), 
    they should win more often. Compare accuracy of (Winner==1) 
    vs (Winner==0) against this heuristic.
    
    Returns:
        1 if Winner==1 means player1 win, else 0
    """
    t = df.dropna(subset=["Ranking_1", "Ranking_2", "Winner"]).copy()
    if len(t) < 1000:
        return 1  # fallback
    
    heuristic = (t["Ranking_1"] < t["Ranking_2"]).astype(int)
    acc_if_winner1 = ((t["Winner"].astype(int) == 1).astype(int) == heuristic).mean()
    acc_if_winner0 = ((t["Winner"].astype(int) == 0).astype(int) == heuristic).mean()
    
    return 1 if acc_if_winner1 >= acc_if_winner0 else 0


def clean_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove post-match leakage columns."""
    drop_cols = [c for c in df.columns if LEAK_RE.search(c)]
    return df.drop(columns=drop_cols, errors="ignore")


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables with safe mappings."""
    df = df.copy()
    
    if "tournament_level" in df.columns:
        df["tournament_level"] = df["tournament_level"].map(LEVEL_MAP).fillna(-1).astype(float)
    if "tournament_surface" in df.columns:
        df["tournament_surface"] = df["tournament_surface"].map(SURFACE_MAP).fillna(-1).astype(float)
    if "round" in df.columns:
        df["round"] = df["round"].map(ROUND_MAP).fillna(-1).astype(float)
    
    return df


def existing_cols(cols: List[str], df: pd.DataFrame) -> List[str]:
    """Return only columns that exist in the dataframe."""
    return [c for c in cols if c in df.columns]


def align_on_ids(res_a: dict, res_b: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Align two result dicts (from evaluate) onto their common match ids.
    
    Returns:
        Tuple of (y_common, p_a_common, p_b_common, common_ids)
    """
    ids_a = res_a["ids"]
    ids_b = res_b["ids"]
    
    pos_a = {mid: i for i, mid in enumerate(ids_a)}
    pos_b = {mid: i for i, mid in enumerate(ids_b)}
    
    common = sorted(set(pos_a).intersection(pos_b))
    if len(common) == 0:
        raise ValueError("No common matches found for paired bootstrap.")
    
    ia = np.array([pos_a[mid] for mid in common], dtype=int)
    ib = np.array([pos_b[mid] for mid in common], dtype=int)
    
    y_a = res_a["y"][ia]
    y_b = res_b["y"][ib]
    if not np.array_equal(y_a, y_b):
        raise ValueError("Label mismatch after alignment.")
    
    return y_a, res_a["p"][ia], res_b["p"][ib], np.array(common, dtype=object)


def objective(brier: float, acc: float, alpha: float = 0.75) -> float:
    """
    Calculate combined objective (Brier-first).
    
    Args:
        brier: Brier score (lower is better)
        acc: Accuracy (higher is better)
        alpha: Weight for Brier score
        
    Returns:
        Combined objective (lower is better)
    """
    return alpha * brier + (1 - alpha) * (1 - acc)