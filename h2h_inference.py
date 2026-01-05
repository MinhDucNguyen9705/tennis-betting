"""
H2H Inference Module

Load pretrained models and predict match outcomes for head-to-head comparisons.
Fetches player features from the database and combines with match context.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db import sql_df, existing_cols

# Try to import joblib for model loading
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# Pretrained models directory
PRETRAINED_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'pretrained_models'
)

# Encoding mappings (same as backtest_pipeline.py)
ENCODINGS = {
    'level': {
        'G': 5,  # Grand Slam
        'M': 4,  # Masters
        'A': 3,  # ATP 500
        'B': 2,  # ATP 250
        'C': 1,  # Challenger
        'F': 0,  # Futures
    },
    'surface': {
        'Hard': 0,
        'Clay': 1,
        'Grass': 2,
        'Carpet': 3,
    },
    'round': {
        'R128': 0,
        'R64': 1,
        'R32': 2,
        'R16': 3,
        'QF': 4,
        'SF': 5,
        'F': 6,
        'Q1': -3,
        'Q2': -2,
        'Q3': -1,
    }
}

# Available pretrained models
AVAILABLE_MODELS = {
    'catboost': 'CatBoost',
    'hist_gradient_boosting': 'Hist Gradient Boosting',
    'logistic_regression': 'Logistic Regression',
    'random_forest': 'Random Forest',
    'svm': 'SVM',
}

# Cached model bundles
_model_cache = {}

# ELO configuration
ELO_BASE = 1500.0
ELO_K = 32.0

# Cached ELO ratings - computed once from all historical matches
_elo_cache = {
    'elo': {},           # player_id -> overall ELO rating
    'surf_elo': {},      # (player_id, surface_code) -> surface-specific ELO
    'rankings': {},      # player_id -> latest ranking
    'computed': False,   # Whether ELO has been computed
}


def list_pretrained_models():
    """List available pretrained models."""
    models = []
    
    if not os.path.exists(PRETRAINED_MODELS_DIR):
        return models
    
    for filename in os.listdir(PRETRAINED_MODELS_DIR):
        if not filename.endswith('.joblib'):
            continue
        
        # Parse filename: {model_type}_{start_year}_{end_year}.joblib
        parts = filename.replace('.joblib', '').split('_')
        if len(parts) >= 3:
            try:
                end_year = int(parts[-1])
                start_year = int(parts[-2])
                model_type = '_'.join(parts[:-2])
                
                display_name = AVAILABLE_MODELS.get(model_type, model_type)
                models.append({
                    'model_type': model_type,
                    'display_name': f"{display_name} ({start_year}-{end_year})",
                    'filename': filename,
                    'filepath': os.path.join(PRETRAINED_MODELS_DIR, filename),
                    'year_range': f"{start_year}-{end_year}",
                })
            except (ValueError, IndexError):
                continue
    
    return sorted(models, key=lambda x: x['display_name'])


def load_model(model_type: str):
    """Load a pretrained model bundle (with caching)."""
    if model_type in _model_cache:
        return _model_cache[model_type]
    
    if not HAS_JOBLIB:
        print("WARNING: joblib not available")
        return None
    
    # Find the model file
    models = list_pretrained_models()
    model_info = next((m for m in models if m['model_type'] == model_type), None)
    
    if model_info is None:
        print(f"WARNING: Model not found: {model_type}")
        return None
    
    try:
        bundle = joblib.load(model_info['filepath'])
        _model_cache[model_type] = bundle
        return bundle
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None


def compute_all_elo_ratings():
    """
    Compute ELO ratings for all players from historical matches.
    
    This runs once and caches the results. ELO is computed chronologically
    to ensure proper temporal ordering.
    """
    global _elo_cache
    
    if _elo_cache['computed']:
        return
    
    print("[H2H Inference] Computing ELO ratings from historical matches...")
    start_time = time.time()
    
    # Query all matches with required columns, sorted by date
    q = """
    SELECT 
        ID_1, ID_2, 
        tournament_surface,
        Winner,
        Ranking_1, Ranking_2,
        CAST(tournament_date AS DATE) AS match_date
    FROM matches
    WHERE ID_1 IS NOT NULL 
      AND ID_2 IS NOT NULL
      AND Winner IS NOT NULL
    ORDER BY match_date ASC
    """
    
    df = sql_df(q)
    
    if df.empty:
        print("[H2H Inference] WARNING: No matches found for ELO computation")
        _elo_cache['computed'] = True
        return
    
    # Initialize ELO dictionaries with default values
    from collections import defaultdict
    elo = defaultdict(lambda: ELO_BASE)
    surf_elo = defaultdict(lambda: ELO_BASE)
    rankings = {}
    
    def expected_score(r1, r2):
        """Calculate expected score using ELO formula."""
        return 1.0 / (1.0 + 10 ** ((r2 - r1) / 400.0))
    
    # Iterate through matches chronologically
    for _, row in df.iterrows():
        p1, p2 = row['ID_1'], row['ID_2']
        surf = row.get('tournament_surface', 'Hard')
        winner = row['Winner']
        
        # Map surface to code
        surf_code = ENCODINGS['surface'].get(surf, 0)
        
        # Current ELO ratings (before this match)
        r1, r2 = elo[p1], elo[p2]
        sr1, sr2 = surf_elo[(p1, surf_code)], surf_elo[(p2, surf_code)]
        
        # Determine actual outcome (1 if P1 wins, 0 if P2 wins)
        # Winner == 0 means P1 wins in most datasets
        actual = 1.0 if winner == 0 else 0.0
        
        # Update overall ELO
        exp1 = expected_score(r1, r2)
        elo[p1] = r1 + ELO_K * (actual - exp1)
        elo[p2] = r2 + ELO_K * ((1 - actual) - (1 - exp1))
        
        # Update surface-specific ELO
        exp_s1 = expected_score(sr1, sr2)
        surf_elo[(p1, surf_code)] = sr1 + ELO_K * (actual - exp_s1)
        surf_elo[(p2, surf_code)] = sr2 + ELO_K * ((1 - actual) - (1 - exp_s1))
        
        # Track latest rankings
        if pd.notna(row.get('Ranking_1')):
            rankings[p1] = int(row['Ranking_1'])
        if pd.notna(row.get('Ranking_2')):
            rankings[p2] = int(row['Ranking_2'])
    
    # Store in cache (convert defaultdict to regular dict)
    _elo_cache['elo'] = dict(elo)
    _elo_cache['surf_elo'] = dict(surf_elo)
    _elo_cache['rankings'] = rankings
    _elo_cache['computed'] = True
    
    elapsed = time.time() - start_time
    print(f"[H2H Inference] ELO computed for {len(elo)} players in {elapsed:.2f}s")


def get_player_elo(player_id, surface=None):
    """
    Get ELO rating for a player.
    
    Args:
        player_id: Player ID
        surface: Optional surface name ('Hard', 'Clay', 'Grass')
                 If provided, returns surface-specific ELO
    
    Returns:
        ELO rating (float), or ELO_BASE if player not found
    """
    # Ensure ELO is computed
    if not _elo_cache['computed']:
        compute_all_elo_ratings()
    
    # Convert player_id to native Python type
    if hasattr(player_id, 'item'):
        player_id = player_id.item()
    else:
        player_id = int(player_id) if isinstance(player_id, (np.integer, np.int64)) else player_id
    
    if surface:
        surf_code = ENCODINGS['surface'].get(surface, 0)
        return _elo_cache['surf_elo'].get((player_id, surf_code), ELO_BASE)
    
    return _elo_cache['elo'].get(player_id, ELO_BASE)


def get_player_ranking(player_id):
    """Get last known ranking for a player."""
    if not _elo_cache['computed']:
        compute_all_elo_ratings()
    
    if hasattr(player_id, 'item'):
        player_id = player_id.item()
    else:
        player_id = int(player_id) if isinstance(player_id, (np.integer, np.int64)) else player_id
    
    return _elo_cache['rankings'].get(player_id, None)


def elo_win_probability(elo1, elo2):
    """
    Calculate win probability using ELO ratings.
    
    Args:
        elo1: Player 1's ELO rating
        elo2: Player 2's ELO rating
    
    Returns:
        Probability that player 1 wins (0-1)
    """
    return 1.0 / (1.0 + 10 ** ((elo2 - elo1) / 400.0))


def ranking_win_probability(rank1, rank2):
    """
    Estimate win probability from ATP rankings.
    
    Uses log-scale comparison since ranking differences matter more at the top.
    
    Args:
        rank1: Player 1's ATP ranking
        rank2: Player 2's ATP ranking
    
    Returns:
        Probability that player 1 wins (0-1)
    """
    if rank1 is None or rank2 is None:
        return 0.5  # No data, 50-50
    
    if rank1 <= 0 or rank2 <= 0:
        return 0.5
    
    # Log-scale comparison
    log_ratio = np.log(rank2 / rank1)
    return 1.0 / (1.0 + np.exp(-0.4 * log_ratio))


def get_player_features(player_id):
    """
    Get the latest available features for a player from the database.
    
    Returns a dict with player stats from their most recent match.
    """
    # Convert to native Python type to avoid DuckDB numpy type issues
    if hasattr(player_id, 'item'):
        player_id = player_id.item()  # numpy scalar to Python
    else:
        player_id = int(player_id) if isinstance(player_id, (np.integer, np.int64)) else player_id
    
    cols = existing_cols()
    
    # List of features we need for the player
    player_stat_cols = [
        'Ranking', 'Ranking_Points', 'Best_Rank', 'Birth_Year', 'Height',
        'Victories_Percentage', 'Clay_Victories_Percentage', 'Grass_Victories_Percentage',
        'Hard_Victories_Percentage', 'Carpet_Victories_Percentage',
        'Aces_Percentage', 'Doublefaults_Percentage',
        'First_Serve_Success_Percentage', 'Winning_on_1st_Serve_Percentage',
        'Winning_on_2nd_Serve_Percentage', 'Overall_Win_on_Serve_Percentage',
        'BreakPoint_Face_Percentage', 'BreakPoint_Saved_Percentage',
    ]
    
    # Build select list checking which columns exist
    select_cols = []
    for col in player_stat_cols:
        c1 = f"{col}_1"
        c2 = f"{col}_2"
        if c1 in cols:
            select_cols.append(c1)
        if c2 in cols:
            select_cols.append(c2)
    
    # Query for player's most recent match
    q = f"""
    WITH player_matches AS (
        SELECT 
            ID_1, ID_2,
            CAST(tournament_date AS DATE) AS match_date,
            {", ".join(select_cols)}
        FROM matches
        WHERE ID_1 = $p OR ID_2 = $p
        ORDER BY match_date DESC
        LIMIT 1
    )
    SELECT * FROM player_matches
    """
    
    df = sql_df(q, {"p": player_id})
    
    if df.empty:
        return None
    
    row = df.iloc[0]
    
    # Extract features from the correct side (1 or 2)
    is_player1 = row.get('ID_1') == player_id
    suffix = '_1' if is_player1 else '_2'
    
    features = {}
    for col in player_stat_cols:
        full_col = f"{col}{suffix}"
        if full_col in row:
            features[col] = row[full_col]
        else:
            features[col] = None
    
    return features


def _parse_birth_year(val):
    """Convert Birth_Year value to integer year (handles both date strings and integers)."""
    if val is None:
        return None
    
    # If already numeric
    if isinstance(val, (int, float)):
        if not pd.isna(val):
            return int(val)
        return None
    
    # If string like '1987-05-22', extract year
    if isinstance(val, str):
        try:
            dt = pd.to_datetime(val, errors='coerce')
            if pd.notna(dt):
                return dt.year
            # Try parsing as just a year
            return int(val)
        except:
            return None
    
    # If datetime-like
    try:
        dt = pd.to_datetime(val, errors='coerce')
        if pd.notna(dt):
            return dt.year
    except:
        pass
    
    return None


def build_match_features(p1_features, p2_features, surface, level, round_val, best_of):
    """
    Build a feature vector for a hypothetical match.
    
    Combines two players' features with match context into model-ready format.
    """
    features = {}
    
    # --- Player 1 raw features ---
    for key, val in (p1_features or {}).items():
        # Convert Birth_Year to integer
        if key == 'Birth_Year':
            val = _parse_birth_year(val)
        features[f"{key}_1"] = val
    
    # --- Player 2 raw features ---
    for key, val in (p2_features or {}).items():
        # Convert Birth_Year to integer
        if key == 'Birth_Year':
            val = _parse_birth_year(val)
        features[f"{key}_2"] = val
    
    # --- Match context ---
    features['tournament_surface'] = ENCODINGS['surface'].get(surface, -1)
    features['tournament_level'] = ENCODINGS['level'].get(level, -1)
    features['round'] = ENCODINGS['round'].get(round_val, -1)
    features['best_of'] = best_of
    
    # --- Engineered features (differences) ---
    def safe_diff(col):
        v1 = features.get(f"{col}_1")
        v2 = features.get(f"{col}_2")
        if v1 is not None and v2 is not None:
            try:
                return float(v1) - float(v2)
            except (ValueError, TypeError):
                return None
        return None
    
    def safe_ratio(col):
        v1 = features.get(f"{col}_1")
        v2 = features.get(f"{col}_2")
        if v1 is not None and v2 is not None:
            try:
                v2_float = float(v2)
                if v2_float == 0:
                    return None
                return float(v1) / v2_float
            except (ValueError, TypeError):
                return None
        return None
    
    # Rank difference (Ranking_2 - Ranking_1 since lower rank = better)
    r1 = features.get('Ranking_1')
    r2 = features.get('Ranking_2')
    if r1 is not None and r2 is not None:
        try:
            features['Rank_Diff'] = float(r2) - float(r1)
            if float(r2) != 0:
                features['Rank_Ratio'] = float(r1) / float(r2)
        except:
            pass
    
    # Points difference
    features['Pts_Diff'] = safe_diff('Ranking_Points')
    features['Pts_Ratio'] = safe_ratio('Ranking_Points')
    
    # Age difference (Birth_Year_2 - Birth_Year_1)
    by1 = features.get('Birth_Year_1')
    by2 = features.get('Birth_Year_2')
    if by1 is not None and by2 is not None:
        try:
            features['Age_Diff'] = float(by2) - float(by1)
        except:
            pass
    
    # Height difference
    features['Height_Diff'] = safe_diff('Height')
    
    # Win percentage differences
    features['WinPct_Diff'] = safe_diff('Victories_Percentage')
    features['Clay_WinPct_Diff'] = safe_diff('Clay_Victories_Percentage')
    features['Grass_WinPct_Diff'] = safe_diff('Grass_Victories_Percentage')
    features['Hard_WinPct_Diff'] = safe_diff('Hard_Victories_Percentage')
    
    # Serve stats differences
    features['Aces_Percentage_Diff'] = safe_diff('Aces_Percentage')
    features['First_Serve_Success_Percentage_Diff'] = safe_diff('First_Serve_Success_Percentage')
    features['Winning_on_1st_Serve_Percentage_Diff'] = safe_diff('Winning_on_1st_Serve_Percentage')
    features['BreakPoint_Saved_Percentage_Diff'] = safe_diff('BreakPoint_Saved_Percentage')
    
    # Elo placeholder (would need historical computation; use ranking-based proxy)
    rp1 = features.get('Ranking_Points_1')
    rp2 = features.get('Ranking_Points_2')
    if rp1 is not None and rp2 is not None:
        try:
            # Simple Elo-like proxy based on ranking points
            features['Elo_1'] = 1500 + float(rp1) / 10
            features['Elo_2'] = 1500 + float(rp2) / 10
            features['Elo_Diff'] = features['Elo_1'] - features['Elo_2']
        except:
            pass
    
    return features


def predict_h2h(model_type, p1_id, p2_id, surface='Hard', level='M', round_val='R32', best_of=3):
    """
    Predict the probability of Player 1 winning against Player 2.
    
    Uses a hybrid approach combining ML model predictions with ELO ratings.
    Falls back gracefully when data is missing:
    - Full ML + ELO hybrid when all features available
    - ELO-only when player features missing
    - Ranking-based when ELO unavailable
    - 50% when no data at all
    
    Args:
        model_type: Pretrained model type (e.g., 'catboost', 'hist_gradient_boosting')
        p1_id: Player 1 ID
        p2_id: Player 2 ID
        surface: 'Hard', 'Clay', or 'Grass'
        level: Tournament level ('G', 'M', 'A', 'B', 'C', 'F')
        round_val: Round ('R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F')
        best_of: Number of sets (3 or 5)
    
    Returns:
        dict: {
            'probability': float (0-1),
            'method': str ('ML+ELO', 'ELO', 'RANKING', 'DEFAULT'),
            'ml_prob': float or None,
            'elo_prob': float or None,
            'surf_elo_prob': float or None
        }
        Returns simple float for backward compatibility if prediction succeeds via ML
    """
    start_time = time.time()
    
    # Ensure ELO ratings are computed
    if not _elo_cache['computed']:
        compute_all_elo_ratings()
    
    # Get ELO-based probabilities
    elo1 = get_player_elo(p1_id)
    elo2 = get_player_elo(p2_id)
    surf_elo1 = get_player_elo(p1_id, surface)
    surf_elo2 = get_player_elo(p2_id, surface)
    
    elo_prob = elo_win_probability(elo1, elo2)
    surf_elo_prob = elo_win_probability(surf_elo1, surf_elo2)
    
    # Check if players have ELO data (not just default base)
    has_p1_elo = elo1 != ELO_BASE
    has_p2_elo = elo2 != ELO_BASE
    has_elo_data = has_p1_elo or has_p2_elo
    
    # Get rankings for fallback
    rank1 = get_player_ranking(p1_id)
    rank2 = get_player_ranking(p2_id)
    ranking_prob = ranking_win_probability(rank1, rank2)
    
    # Load ML model
    bundle = load_model(model_type)
    ml_prob = None
    
    if bundle is not None:
        model = bundle.get('model')
        feature_names = bundle.get('features', [])
        
        if model is not None:
            # Get player features
            p1_features = get_player_features(p1_id)
            p2_features = get_player_features(p2_id)
            
            if p1_features is not None and p2_features is not None:
                # Build match features
                match_features = build_match_features(
                    p1_features, p2_features, surface, level, round_val, best_of
                )
                
                # Create feature vector
                X = []
                for fname in feature_names:
                    val = match_features.get(fname, -1)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        val = -1
                    X.append(val)
                
                X = np.array([X])
                
                try:
                    proba = model.predict_proba(X)
                    ml_prob = float(proba[0, 0])
                except Exception as e:
                    print(f"[H2H Inference] ML prediction error: {e}")
                    ml_prob = None
    
    # Determine final probability and method
    if ml_prob is not None:
        # Hybrid: Weight surface ELO more heavily since it captures surface-specific form
        # ML model already includes these features but with lower importance
        # Base weights: 60% ML + 25% surface ELO + 15% overall ELO
        ml_weight = 0.60
        surf_elo_weight = 0.25
        elo_weight = 0.15
        
        final_prob = ml_weight * ml_prob + surf_elo_weight * surf_elo_prob + elo_weight * elo_prob
        method = 'ML+ELO'
    elif has_elo_data:
        # ELO only: 70% surface ELO + 30% overall ELO (surface matters more!)
        final_prob = 0.70 * surf_elo_prob + 0.30 * elo_prob
        method = 'ELO'
    elif rank1 is not None or rank2 is not None:
        # Ranking-based fallback
        final_prob = ranking_prob
        method = 'RANKING'
    else:
        # No data at all - 50/50
        final_prob = 0.5
        method = 'DEFAULT'
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    # Log prediction info
    print(f"[H2H Inference] Method: {method} | Model: {model_type} | "
          f"ELO: {elo1:.0f} vs {elo2:.0f} | SurfELO: {surf_elo1:.0f} vs {surf_elo2:.0f} | "
          f"{surface}/{level}/{round_val}/BO{best_of} | "
          f"P1 Win: {final_prob*100:.1f}% | Time: {elapsed_ms:.1f}ms")
    
    # Return dict with all info for transparency
    return {
        'probability': float(final_prob),
        'method': method,
        'ml_prob': ml_prob,
        'elo_prob': float(elo_prob),
        'surf_elo_prob': float(surf_elo_prob),
        'ranking_prob': float(ranking_prob) if rank1 or rank2 else None,
        'p1_elo': float(elo1),
        'p2_elo': float(elo2),
        'p1_surf_elo': float(surf_elo1),
        'p2_surf_elo': float(surf_elo2),
    }


def get_model_options():
    """Get dropdown options for the model selector."""
    models = list_pretrained_models()
    if not models:
        return [{"label": "No models available", "value": "none"}]
    
    return [{"label": m['display_name'], "value": m['model_type']} for m in models]


def get_surface_options():
    """Get dropdown options for surface."""
    return [
        {"label": "Hard", "value": "Hard"},
        {"label": "Clay", "value": "Clay"},
        {"label": "Grass", "value": "Grass"},
    ]


def get_level_options():
    """Get dropdown options for tournament level."""
    return [
        {"label": "Grand Slam", "value": "G"},
        {"label": "Masters", "value": "M"},
        {"label": "ATP 500", "value": "A"},
        {"label": "ATP 250", "value": "B"},
        {"label": "Challenger", "value": "C"},
        {"label": "Futures", "value": "F"},
    ]


def get_round_options():
    """Get dropdown options for round."""
    return [
        {"label": "R128", "value": "R128"},
        {"label": "R64", "value": "R64"},
        {"label": "R32", "value": "R32"},
        {"label": "R16", "value": "R16"},
        {"label": "Quarter-Final", "value": "QF"},
        {"label": "Semi-Final", "value": "SF"},
        {"label": "Final", "value": "F"},
    ]
