# =========================
# Configuration Settings
# =========================

import re

DATA_PATH = "/kaggle/input/match-with-odds-2010-2024"
TRAIN_YEARS = [2019, 2020, 2021, 2022]
CALIB_YEARS = [2023]
TEST_YEARS = [2024]
CSV_SEP = ";"

# Model parameters
ALPHA = 0.75  # objective = alpha*Brier + (1-alpha)*(1-Acc)  (Brier-first)
THRESHOLD = 0.5
RECENT_K = 10  # For "recent form" features from match-history strings

# Elo parameters
ELO_BASE = 1500.0
ELO_K = 32.0

# Encoding mappings
LEVEL_MAP = {"G": 4, "M": 3, "A": 2, "C": 1, "F": 0}
SURFACE_MAP = {"Hard": 3, "Clay": 2, "Grass": 1, "Carpet": 0}
ROUND_MAP = {
    "F": 6, "SF": 5, "QF": 4, "R16": 3, "R32": 2, 
    "R64": 1, "R128": 0, "Q1": -3, "Q2": -2, "Q3": -1
}

# Leakage patterns (post-match data that shouldn't be used for prediction)
LEAK_PATTERNS = [
    r"^score$", r"elapsed_minutes",
    r"^aces_nb_", r"^doublefaults_nb_",
    r"^svpt_", r"^1stIn_", r"^1stWon_", r"^2ndWon_",
    r"^SvGms_", r"^bpSaved_", r"^bpFaced_",
]
LEAK_RE = re.compile("|".join(LEAK_PATTERNS))

# Odds columns
ODDS_COLS = ["B365_1", "B365_2", "PS_1", "PS_2", "Max_1", "Max_2", "Avg_1", "Avg_2"]

# Feature sets for ablation study
FEATURE_SETS_CONFIG = {
    "BASE": [
        "Rank_Diff", "Pts_Diff",
        "tournament_surface", "tournament_level", "round", "best_of"
    ],
    "SURFACE_FORM": "BASE + surface_win_pct",
    "RECENT_FORM": "SURFACE_FORM + recent_form",
    "FATIGUE": "RECENT_FORM + fatigue",
    "LEFTY": "FATIGUE + handedness",
    "ELO_BLOCK": "LEFTY + elo"
}