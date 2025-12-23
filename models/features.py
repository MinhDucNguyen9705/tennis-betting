# =========================
# Feature Engineering
# =========================

import numpy as np
import pandas as pd
from collections import defaultdict
from utils import safe_to_datetime, recent_win_rate
from config import RECENT_K


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add core differential features."""
    df = df.copy()
    
    # Core diffs (very strong for calibration/Brier)
    for a, b, new in [
        ("Ranking_1", "Ranking_2", "Rank_Diff"),
        ("Ranking_Points_1", "Ranking_Points_2", "Pts_Diff"),
        ("Victories_Percentage_1", "Victories_Percentage_2", "WinPct_Diff"),
    ]:
        if a in df.columns and b in df.columns:
            df[new] = df[a] - df[b]
    
    return df


def add_surface_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add surface-specific win percentage diffs."""
    df = df.copy()
    
    for surf in ["Hard", "Clay", "Grass", "Carpet"]:
        a = f"{surf}_Victories_Percentage_1"
        b = f"{surf}_Victories_Percentage_2"
        if a in df.columns and b in df.columns:
            df[f"{surf}_WinPct_Diff"] = df[a] - df[b]
    
    return df


def add_recent_form_features(df: pd.DataFrame, k: int = RECENT_K) -> pd.DataFrame:
    """Add recent form features from match history strings."""
    df = df.copy()
    
    for surf in ["Hard", "Clay", "Grass", "Carpet"]:
        c1 = f"Matches_{surf}_1"
        c2 = f"Matches_{surf}_2"
        if c1 in df.columns and c2 in df.columns:
            df[f"{surf}_RecentWin_1"] = df[c1].apply(lambda x: recent_win_rate(x, k=k))
            df[f"{surf}_RecentWin_2"] = df[c2].apply(lambda x: recent_win_rate(x, k=k))
            df[f"{surf}_RecentWin_Diff"] = df[f"{surf}_RecentWin_1"] - df[f"{surf}_RecentWin_2"]
    
    return df


def add_fatigue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add fatigue and rust features."""
    df = df.copy()
    
    # Games and minutes fatigue
    if "games_fatigue_1" in df.columns and "games_fatigue_2" in df.columns:
        df["GamesFatigue_Diff"] = df["games_fatigue_1"] - df["games_fatigue_2"]
    if "minutes_fatigue_1" in df.columns and "minutes_fatigue_2" in df.columns:
        df["MinutesFatigue_Diff"] = df["minutes_fatigue_1"] - df["minutes_fatigue_2"]
    
    # Date conversions
    if "Last_Tournament_Date_1" in df.columns:
        df["Last_Tournament_Date_1"] = safe_to_datetime(df["Last_Tournament_Date_1"])
    if "Last_Tournament_Date_2" in df.columns:
        df["Last_Tournament_Date_2"] = safe_to_datetime(df["Last_Tournament_Date_2"])
    
    # Days since last tournament
    if "tournament_date" in df.columns:
        if "Last_Tournament_Date_1" in df.columns:
            df["DaysSinceLast_1"] = (df["tournament_date"] - df["Last_Tournament_Date_1"]).dt.days
        if "Last_Tournament_Date_2" in df.columns:
            df["DaysSinceLast_2"] = (df["tournament_date"] - df["Last_Tournament_Date_2"]).dt.days
        if "DaysSinceLast_1" in df.columns and "DaysSinceLast_2" in df.columns:
            df["DaysSinceLast_Diff"] = df["DaysSinceLast_1"] - df["DaysSinceLast_2"]
    
    return df


def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add age-related features."""
    df = df.copy()
    
    if all(c in df.columns for c in ["Birth_Year_1", "Birth_Year_2", "tournament_date"]):
        b1 = safe_to_datetime(df["Birth_Year_1"])
        b2 = safe_to_datetime(df["Birth_Year_2"])
        df["Age_1"] = (df["tournament_date"] - b1).dt.days / 365.25
        df["Age_2"] = (df["tournament_date"] - b2).dt.days / 365.25
        df["Age_Diff"] = df["Age_1"] - df["Age_2"]
    
    return df


def add_handedness_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add handedness and lefty matchup features."""
    df = df.copy()
    
    if "Hand_1" in df.columns and "Hand_2" in df.columns:
        df["IsLeft_1"] = (df["Hand_1"] == "L").astype(int)
        df["IsLeft_2"] = (df["Hand_2"] == "L").astype(int)
        df["LeftyMatchup"] = ((df["Hand_1"] == "L") & (df["Hand_2"] == "R")).astype(int)
    
    return df


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transformations."""
    df = add_basic_features(df)
    df = add_surface_features(df)
    df = add_recent_form_features(df)
    df = add_fatigue_features(df)
    df = add_age_features(df)
    df = add_handedness_features(df)
    return df


def compute_elo_features(df: pd.DataFrame, base: float = 1500.0, k: float = 32.0) -> pd.DataFrame:
    """
    Compute Elo and Surface-Elo features (dynamic strength).
    
    Args:
        df: DataFrame with match data (must be sorted by tournament_date)
        base: Initial Elo rating
        k: Elo K-factor
        
    Returns:
        DataFrame with Elo features added
    """
    dd = df.copy()
    
    if "tournament_date" not in dd.columns:
        raise ValueError("Need tournament_date for Elo ordering.")
    
    dd = dd.sort_values("tournament_date")
    
    elo = defaultdict(lambda: base)
    selo = defaultdict(lambda: base)  # (player_id, surface)
    
    Elo1, Elo2, SElo1, SElo2 = [], [], [], []
    
    def expected(a, b):
        return 1.0 / (1.0 + 10**((b - a) / 400.0))
    
    for _, row in dd.iterrows():
        p1, p2 = row.get("ID_1"), row.get("ID_2")
        surf_code = row.get("tournament_surface")
        
        r1, r2 = elo[p1], elo[p2]
        sr1, sr2 = selo[(p1, surf_code)], selo[(p2, surf_code)]
        
        Elo1.append(r1)
        Elo2.append(r2)
        SElo1.append(sr1)
        SElo2.append(sr2)
        
        y = row.get("y")
        if pd.isna(y):
            continue
        
        # Update overall Elo
        ea = expected(r1, r2)
        elo[p1] = r1 + k * (y - ea)
        elo[p2] = r2 + k * ((1 - y) - (1 - ea))
        
        # Update surface-specific Elo
        esa = expected(sr1, sr2)
        selo[(p1, surf_code)] = sr1 + k * (y - esa)
        selo[(p2, surf_code)] = sr2 + k * ((1 - y) - (1 - esa))
    
    dd["Elo_1"], dd["Elo_2"] = Elo1, Elo2
    dd["EloDiff"] = dd["Elo_1"] - dd["Elo_2"]
    dd["SurfElo_1"], dd["SurfElo_2"] = SElo1, SElo2
    dd["SurfEloDiff"] = dd["SurfElo_1"] - dd["SurfElo_2"]
    
    return dd


def get_feature_sets(df: pd.DataFrame) -> dict:
    """
    Define feature sets for ablation study.
    
    Returns:
        Dictionary mapping feature set name to list of column names
    """
    BASE = [
        "Rank_Diff", "Pts_Diff",
        "tournament_surface", "tournament_level", "round", "best_of"
    ]
    
    SURFACE_FORM = BASE + [
        "Hard_WinPct_Diff", "Clay_WinPct_Diff", "Grass_WinPct_Diff"
    ]
    
    RECENT_FORM = SURFACE_FORM + [
        "Hard_RecentWin_Diff", "Clay_RecentWin_Diff", "Grass_RecentWin_Diff"
    ]
    
    FATIGUE = RECENT_FORM + [
        "GamesFatigue_Diff", "MinutesFatigue_Diff", "DaysSinceLast_Diff"
    ]
    
    LEFTY = FATIGUE + [
        "IsLeft_1", "IsLeft_2", "LeftyMatchup"
    ]
    
    ELO_BLOCK = LEFTY + [
        "EloDiff", "SurfEloDiff"
    ]
    
    return {
        "BASE": BASE,
        "SURFACE_FORM": SURFACE_FORM,
        "RECENT_FORM": RECENT_FORM,
        "FATIGUE": FATIGUE,
        "LEFTY": LEFTY,
        "ELO_BLOCK": ELO_BLOCK,
    }