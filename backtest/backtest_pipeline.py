import os
import sys
import re
import ast
import logging
from datetime import datetime
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

# Try to import sklearn for model training
try:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Try to import CatBoost
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Model types available
MODEL_TYPES = {
    'hist_gradient_boosting': 'HistGradientBoosting',
    'random_forest': 'RandomForest',
    'logistic_regression': 'Logistic Regression',
    'catboost': 'CatBoost',
}

# Default data directory
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data_tennis_match_reduced'
)

# Available year ranges
TRAINING_YEARS_AVAILABLE = list(range(2000, 2025))  # 2000-2024
BACKTEST_YEARS_AVAILABLE = list(range(2010, 2025))  # 2010-2024 (with odds)

# Default feature encodings
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

# Leakage patterns (post-match data that shouldn't be used for prediction)
LEAK_PATTERNS = [
    r"^score$", r"elapsed_minutes",
    r"^aces_nb_", r"^doublefaults_nb_",
    r"^svpt_", r"^1stIn_", r"^1stWon_", r"^2ndWon_",
    r"^SvGms_", r"^bpSaved_", r"^bpFaced_",
]
LEAK_RE = re.compile("|".join(LEAK_PATTERNS))


def get_logger():
    """Get or create logger."""
    logger = logging.getLogger('backtest_pipeline')
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def validate_no_leakage(train_years: List[int], backtest_start_date: str) -> Tuple[bool, str]:
    """
    Validate that there is no data leakage between training and backtest periods.
    
    Args:
        train_years: List of years used for training
        backtest_start_date: Start date of backtest period (YYYY-MM-DD)
        
    Returns:
        Tuple of (is_valid, message)
    """
    if not train_years:
        return False, "No training years selected"
    
    max_train_year = max(train_years)
    
    try:
        backtest_start = datetime.strptime(backtest_start_date, '%Y-%m-%d')
    except ValueError:
        return False, f"Invalid date format: {backtest_start_date}"
    
    if backtest_start.year <= max_train_year:
        return False, f"Data leakage detected: Backtest starts in {backtest_start.year} but training includes {max_train_year}"
    
    return True, f"✅ No data leakage: Training ends {max_train_year}, Backtest starts {backtest_start.year}"


def load_training_data(
    years: List[int],
    data_dir: str = None,
    csv_sep: str = ';'
) -> pd.DataFrame:
    """
    Load training data from matches_data_*.csv files.
    
    Args:
        years: List of years to load
        data_dir: Directory containing data files
        csv_sep: CSV separator
        
    Returns:
        Combined DataFrame with all training data
    """
    logger = get_logger()
    
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    all_dfs = []
    
    for year in years:
        file_path = os.path.join(data_dir, f'matches_data_{year}.csv')
        
        if not os.path.exists(file_path):
            logger.warning(f"Training data file not found: {file_path}")
            continue
        
        logger.info(f"Loading training data for {year}...")
        df = pd.read_csv(file_path, sep=csv_sep)
        df['source_year'] = year
        all_dfs.append(df)
        logger.info(f"  Loaded {len(df)} matches from {year}")
    
    if not all_dfs:
        raise FileNotFoundError(f"No training data files found for years: {years}")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Total training matches: {len(combined)}")
    
    return combined


def load_backtest_data(
    start_date: str,
    end_date: str,
    data_dir: str = None,
    csv_sep: str = ';'
) -> pd.DataFrame:
    """
    Load backtest data from matches_with_odds_*.csv files.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        data_dir: Directory containing data files
        csv_sep: CSV separator
        
    Returns:
        DataFrame with matches in date range (with odds)
    """
    logger = get_logger()
    
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Determine which years to load
    years_to_load = list(range(start_dt.year, end_dt.year + 1))
    
    all_dfs = []
    
    for year in years_to_load:
        file_path = os.path.join(data_dir, f'matches_with_odds_{year}.csv')
        
        if not os.path.exists(file_path):
            logger.warning(f"Backtest data file not found: {file_path}")
            continue
        
        logger.info(f"Loading backtest data for {year}...")
        df = pd.read_csv(file_path, sep=csv_sep)
        all_dfs.append(df)
        logger.info(f"  Loaded {len(df)} matches from {year}")
    
    if not all_dfs:
        raise FileNotFoundError(f"No backtest data files found for date range: {start_date} to {end_date}")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Parse dates and filter to exact range
    if 'tournament_date' in combined.columns:
        combined['tournament_date'] = pd.to_datetime(combined['tournament_date'], errors='coerce')
        combined = combined[
            (combined['tournament_date'] >= start_date) &
            (combined['tournament_date'] <= end_date)
        ]
    elif 'match_date' in combined.columns:
        combined['match_date'] = pd.to_datetime(combined['match_date'], errors='coerce')
        combined = combined[
            (combined['match_date'] >= start_date) &
            (combined['match_date'] <= end_date)
        ]
    
    logger.info(f"Total backtest matches in date range: {len(combined)}")
    
    return combined


def clean_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that contain post-match information."""
    leak_cols = [c for c in df.columns if LEAK_RE.search(c)]
    return df.drop(columns=leak_cols, errors='ignore')


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables."""
    d = df.copy()
    
    if 'tournament_level' in d.columns:
        d['tournament_level'] = d['tournament_level'].map(ENCODINGS['level']).fillna(-1)
    if 'tournament_surface' in d.columns:
        d['tournament_surface'] = d['tournament_surface'].map(ENCODINGS['surface']).fillna(-1)
    if 'round' in d.columns:
        d['round'] = d['round'].map(ENCODINGS['round']).fillna(-1)
    
    return d


def recent_win_rate(s, k: int = 10) -> float:
    """Calculate win rate from recent match history string."""
    try:
        lst = ast.literal_eval(s) if isinstance(s, str) else s
        if not isinstance(lst, list) or len(lst) == 0:
            return np.nan
        lst = lst[-k:]
        return sum(1 for x in lst if x == "V") / len(lst)
    except:
        return np.nan


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic engineered features."""
    d = df.copy()
    
    # Helper function to safely convert to numeric and compute difference
    def safe_diff(col1, col2, result_col):
        if col1 in d.columns and col2 in d.columns:
            d[col1] = pd.to_numeric(d[col1], errors='coerce')
            d[col2] = pd.to_numeric(d[col2], errors='coerce')
            d[result_col] = d[col1] - d[col2]
    
    def safe_ratio(col1, col2, result_col):
        if col1 in d.columns and col2 in d.columns:
            d[col1] = pd.to_numeric(d[col1], errors='coerce')
            d[col2] = pd.to_numeric(d[col2], errors='coerce')
            d[result_col] = d[col1] / d[col2].replace(0, np.nan)
    
    # Rank difference
    safe_diff('Ranking_2', 'Ranking_1', 'Rank_Diff')
    safe_ratio('Ranking_1', 'Ranking_2', 'Rank_Ratio')
    
    # Points difference  
    safe_diff('Ranking_Points_1', 'Ranking_Points_2', 'Pts_Diff')
    safe_ratio('Ranking_Points_1', 'Ranking_Points_2', 'Pts_Ratio')
    
    # Age difference
    safe_diff('Birth_Year_2', 'Birth_Year_1', 'Age_Diff')
    
    # Height difference
    safe_diff('Height_1', 'Height_2', 'Height_Diff')
    
    # Win percentage difference
    safe_diff('Victories_Percentage_1', 'Victories_Percentage_2', 'WinPct_Diff')
    
    # Surface-specific win rates
    for suf in ['Clay', 'Grass', 'Hard']:
        col1 = f'{suf}_Victories_Percentage_1'
        col2 = f'{suf}_Victories_Percentage_2'
        safe_diff(col1, col2, f'{suf}_WinPct_Diff')
    
    # Serve stats difference
    for stat in ['Aces_Percentage', 'First_Serve_Success_Percentage', 
                 'Winning_on_1st_Serve_Percentage', 'BreakPoint_Saved_Percentage']:
        col1 = f'{stat}_1'
        col2 = f'{stat}_2'
        safe_diff(col1, col2, f'{stat}_Diff')
    
    # Recent form (if Ranking_History available)
    for i in [1, 2]:
        col = f'Ranking_History_{i}'
        if col in d.columns:
            d[f'recent_form_{i}'] = d[col].apply(recent_win_rate)
    
    if 'recent_form_1' in d.columns and 'recent_form_2' in d.columns:
        d['Recent_Form_Diff'] = d['recent_form_1'] - d['recent_form_2']
    
    return d


def compute_simple_elo(df: pd.DataFrame, base: float = 1500, k: float = 32) -> pd.DataFrame:
    """Compute simple Elo ratings for players."""
    d = df.copy()
    
    # Initialize Elo dictionary
    elo = {}
    elo_1 = []
    elo_2 = []
    
    # Sort by date if available
    if 'tournament_date' in d.columns:
        d = d.sort_values('tournament_date')
    
    for idx, row in d.iterrows():
        p1 = row.get('ID_1', row.get('Name_1', f'p1_{idx}'))
        p2 = row.get('ID_2', row.get('Name_2', f'p2_{idx}'))
        
        # Get current ratings
        r1 = elo.get(p1, base)
        r2 = elo.get(p2, base)
        
        elo_1.append(r1)
        elo_2.append(r2)
        
        # Calculate expected scores
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 - e1
        
        # Determine actual scores
        winner = row.get('Winner', None)
        if pd.notna(winner):
            s1 = 1 if winner == 0 else 0
            s2 = 1 - s1
            
            # Update ratings
            elo[p1] = r1 + k * (s1 - e1)
            elo[p2] = r2 + k * (s2 - e2)
    
    d['Elo_1'] = elo_1
    d['Elo_2'] = elo_2
    d['Elo_Diff'] = d['Elo_1'] - d['Elo_2']
    
    return d


def infer_player1_win_mapping(df: pd.DataFrame) -> int:
    """Auto-infer if Winner==0 or Winner==1 means Player1 wins."""
    t = df.dropna(subset=["Ranking_1", "Ranking_2", "Winner"]).copy()
    if len(t) < 1000:
        return 0  # Default: Winner==0 means P1 wins
    
    heuristic = (t["Ranking_1"] < t["Ranking_2"]).astype(int)
    acc_if_winner0 = ((t["Winner"].astype(int) == 0).astype(int) == heuristic).mean()
    acc_if_winner1 = ((t["Winner"].astype(int) == 1).astype(int) == heuristic).mean()
    
    return 0 if acc_if_winner0 >= acc_if_winner1 else 1


def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare training data with feature engineering."""
    logger = get_logger()
    
    # Clean leakage
    df = clean_leakage_columns(df)
    
    # Encode categorical
    df = encode_categorical(df)
    
    # Infer target
    if 'y' not in df.columns:
        if 'Winner' in df.columns:
            winner_val = infer_player1_win_mapping(df)
            df['y'] = (df['Winner'] == winner_val).astype(int)
        else:
            raise ValueError("Cannot determine target variable 'y'")
    
    # Add features
    logger.info("Engineering features...")
    df = add_basic_features(df)
    
    # Compute Elo
    logger.info("Computing Elo ratings...")
    df = compute_simple_elo(df)
    
    return df


def select_features(df: pd.DataFrame) -> List[str]:
    """Select feature columns automatically."""
    exclude = {'y', 'tournament_date', 'Winner', 'score', 'ID_1', 'ID_2', 
               'source_year', 'match_date', 'Name_1', 'Name_2', 'tournament',
               'match_id', 'id'}
    
    features = []
    for c in df.columns:
        if c in exclude:
            continue
        if c.startswith('Ranking_History'):
            continue
        if c.startswith('Versus_'):
            continue
        if df[c].dtype in [np.float64, np.int64, np.int32, np.float32]:
            features.append(c)
    
    return features


def train_model(
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame = None,
    model_type: str = 'hist_gradient_boosting',
    use_calibration: bool = True,
) -> dict:
    """
    Train model on prepared data.
    
    Args:
        train_df: Training dataframe
        calib_df: Calibration dataframe (optional)
        model_type: One of 'hist_gradient_boosting', 'random_forest', 'logistic_regression', 'catboost'
        use_calibration: Whether to apply probability calibration
        
    Returns:
        Model bundle dict
    """
    logger = get_logger()
    
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for training")
    
    # Select features
    features = select_features(train_df)
    logger.info(f"Selected {len(features)} features")
    
    # Prepare matrices
    X_train = train_df[features].fillna(-1).values
    y_train = train_df['y'].values
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Model type: {MODEL_TYPES.get(model_type, model_type)}")
    
    # Create model based on model_type
    if model_type == 'hist_gradient_boosting':
        base_model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_depth=6,
            min_samples_leaf=20,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42
        )
    elif model_type == 'random_forest':
        base_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
    elif model_type == 'logistic_regression':
        # Use Pipeline with StandardScaler to ensure convergence
        base_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=1.0,
                max_iter=2000,
                solver='lbfgs',
                n_jobs=-1,
                random_state=42
            ))
        ])
    elif model_type == 'catboost':
        if not HAS_CATBOOST:
            raise ImportError("CatBoost is not installed. Install with: pip install catboost")
        base_model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_TYPES.keys())}")
    
    # Train with or without calibration
    if use_calibration and calib_df is not None:
        X_calib = calib_df[features].fillna(-1).values
        y_calib = calib_df['y'].values
        logger.info(f"Calibration set: {len(X_calib)} samples")
        
        # Combine train and calib for calibrated training
        X_combined = np.vstack([X_train, X_calib])
        y_combined = np.concatenate([y_train, y_calib])
        
        # Use CalibratedClassifierCV with cross-validation
        model = CalibratedClassifierCV(base_model, cv=3, method='isotonic')
        model.fit(X_combined, y_combined)
    else:
        base_model.fit(X_train, y_train)
        model = base_model
    
    logger.info("Model training completed")
    
    bundle = {
        'model': model,
        'model_type': model_type,
        'features': features,
        'threshold': 0.5,
        'trained_at': datetime.now().isoformat()
    }
    
    return bundle


def run_inference(
    model_bundle: dict,
    backtest_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run inference on backtest data."""
    logger = get_logger()
    
    model = model_bundle['model']
    features = model_bundle['features']
    threshold = model_bundle['threshold']
    
    # Prepare data (similar to training)
    df = clean_leakage_columns(backtest_df.copy())
    df = encode_categorical(df)
    df = add_basic_features(df)
    df = compute_simple_elo(df)
    
    # Check for missing features
    missing = [c for c in features if c not in df.columns]
    if missing:
        logger.warning(f"Missing features (will use -1): {len(missing)} features")
        for c in missing:
            df[c] = -1
    
    # Predict
    X = df[features].fillna(-1).values
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    
    # Create result dataframe with necessary columns for backtesting
    result = pd.DataFrame()
    
    # Keep essential columns
    keep_cols = [
        'tournament_date', 'tournament', 'tournament_level', 'tournament_surface',
        'round', 'Name_1', 'Name_2', 'Winner', 
        'PS_1', 'PS_2', 'B365_1', 'B365_2',
        'Ranking_1', 'Ranking_2'
    ]
    
    for col in keep_cols:
        if col in backtest_df.columns:
            result[col] = backtest_df[col].values
    
    # Add predictions
    result['Prob_P1'] = proba
    result['pred_player1_win'] = pred
    
    # Transform Winner for backtesting (Victory = 0 if P1 wins)
    if 'Winner' in result.columns:
        result['Victory'] = result['Winner'].values
    
    logger.info(f"Inference completed: {len(result)} matches")
    logger.info(f"  P1 win predictions: {pred.sum()} ({pred.mean()*100:.1f}%)")
    
    return result


def train_model_only(
    train_years: List[int],
    calib_years: List[int] = None,
    data_dir: str = None,
    use_calibration: bool = True,
    model_type: str = 'hist_gradient_boosting',
) -> dict:
    """
    Train model only (without running backtest).
    
    Args:
        train_years: Years for training data
        calib_years: Optional years for calibration (defaults to last year of training)
        data_dir: Data directory
        use_calibration: Whether to use calibration
        model_type: One of 'hist_gradient_boosting', 'random_forest', 'logistic_regression', 'catboost'
        
    Returns:
        model_bundle: Dictionary containing trained model and metadata
    """
    logger = get_logger()
    
    # Split train/calib years if needed
    if calib_years is None and use_calibration and len(train_years) > 1:
        calib_years = [max(train_years)]
        train_years = [y for y in train_years if y not in calib_years]
        logger.info(f"Auto-split: Training {train_years}, Calibration {calib_years}")
    
    # Load training data
    logger.info("Loading training data...")
    train_raw = load_training_data(train_years, data_dir)
    train_df = prepare_training_data(train_raw)
    
    # Load calibration data if needed
    calib_df = None
    if use_calibration and calib_years:
        logger.info("Loading calibration data...")
        calib_raw = load_training_data(calib_years, data_dir)
        calib_df = prepare_training_data(calib_raw)
    
    # Train model
    logger.info(f"Training {MODEL_TYPES.get(model_type, model_type)} model...")
    model_bundle = train_model(train_df, calib_df, model_type=model_type, use_calibration=use_calibration)
    
    # Store training config in bundle
    model_bundle['train_years'] = train_years
    model_bundle['calib_years'] = calib_years
    model_bundle['max_train_year'] = max(train_years) if train_years else None
    
    logger.info("Model training completed")
    
    return model_bundle


def run_backtest_only(
    model_bundle: dict,
    backtest_start_date: str,
    backtest_end_date: str,
    data_dir: str = None,
) -> pd.DataFrame:
    """
    Run backtest inference using a pre-trained model bundle.
    
    This allows changing backtest date ranges without retraining the model.
    
    Args:
        model_bundle: Pre-trained model bundle from train_model_only
        backtest_start_date: Backtest start date (YYYY-MM-DD)
        backtest_end_date: Backtest end date (YYYY-MM-DD)
        data_dir: Data directory
        
    Returns:
        predictions_df: DataFrame with predictions for backtesting
    """
    logger = get_logger()
    
    # Validate no data leakage with the trained model
    max_train_year = model_bundle.get('max_train_year')
    if max_train_year:
        is_valid, message = validate_no_leakage([max_train_year], backtest_start_date)
        if not is_valid:
            raise ValueError(message)
        logger.info(message)
    
    # Load backtest data
    logger.info(f"Loading backtest data: {backtest_start_date} to {backtest_end_date}")
    backtest_raw = load_backtest_data(backtest_start_date, backtest_end_date, data_dir)
    
    # Run inference
    logger.info("Running inference...")
    predictions_df = run_inference(model_bundle, backtest_raw)
    
    return predictions_df


def run_full_pipeline(
    train_years: List[int],
    backtest_start_date: str,
    backtest_end_date: str,
    calib_years: List[int] = None,
    data_dir: str = None,
    use_calibration: bool = True,
) -> Tuple[dict, pd.DataFrame]:
    """
    Run the complete training and inference pipeline.
    
    Args:
        train_years: Years for training data
        backtest_start_date: Backtest start date (YYYY-MM-DD)
        backtest_end_date: Backtest end date (YYYY-MM-DD)
        calib_years: Optional years for calibration (defaults to last year of training)
        data_dir: Data directory
        use_calibration: Whether to use calibration
        
    Returns:
        Tuple of (model_bundle, predictions_df)
    """
    logger = get_logger()
    
    # Validate no data leakage
    is_valid, message = validate_no_leakage(train_years, backtest_start_date)
    if not is_valid:
        raise ValueError(message)
    logger.info(message)
    
    # Train model
    model_bundle = train_model_only(train_years, calib_years, data_dir, use_calibration)
    
    # Run backtest
    predictions_df = run_backtest_only(model_bundle, backtest_start_date, backtest_end_date, data_dir)
    
    return model_bundle, predictions_df
