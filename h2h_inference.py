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
    
    Args:
        model_type: Pretrained model type (e.g., 'catboost', 'hist_gradient_boosting')
        p1_id: Player 1 ID
        p2_id: Player 2 ID
        surface: 'Hard', 'Clay', or 'Grass'
        level: Tournament level ('G', 'M', 'A', 'B', 'C', 'F')
        round_val: Round ('R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F')
        best_of: Number of sets (3 or 5)
    
    Returns:
        float: Probability of Player 1 winning (0-1), or None if prediction fails
    """
    # Load model
    bundle = load_model(model_type)
    if bundle is None:
        return None
    
    model = bundle.get('model')
    feature_names = bundle.get('features', [])
    
    if model is None:
        return None
    
    # Get player features
    p1_features = get_player_features(p1_id)
    p2_features = get_player_features(p2_id)
    
    if p1_features is None or p2_features is None:
        return None
    
    # Build match features
    match_features = build_match_features(p1_features, p2_features, surface, level, round_val, best_of)
    
    # Create feature vector in correct order
    X = []
    for fname in feature_names:
        val = match_features.get(fname, -1)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = -1
        X.append(val)
    
    X = np.array([X])
    
    # Predict
    try:
        start_time = time.time()
        proba = model.predict_proba(X)
        # proba[:, 0] is probability of class 0 (P1 wins when Winner==0)
        p1_win_prob = proba[0, 0]
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Log prediction info
        p1_rank = match_features.get('Ranking_1', '?')
        p2_rank = match_features.get('Ranking_2', '?')
        print(f"[H2H Inference] Model: {model_type} | P1(rank {p1_rank}) vs P2(rank {p2_rank}) | "
              f"{surface}/{level}/{round_val}/BO{best_of} | "
              f"P1 Win: {p1_win_prob*100:.1f}% | Time: {elapsed_ms:.1f}ms")
        
        return float(p1_win_prob)
    except Exception as e:
        print(f"[H2H Inference] Prediction error: {e}")
        return None


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
