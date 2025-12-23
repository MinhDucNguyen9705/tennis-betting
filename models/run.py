# =========================
# Model Runner - Train, Test, Eval, Infer
# =========================

import os
import sys
import yaml
import argparse
import logging
import glob
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Add models directory to path
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from config import LEAK_RE
from utils import load_years, safe_to_datetime, infer_player1_win_mapping
from features import add_all_features, compute_elo_features
from histgradientboosting import HistGradientBoostingModel
from xgboost_model import XGBoostModel
from train_test import paired_bootstrap_delta, print_evaluation_results, print_bootstrap_comparison



def setup_logging(level_str: str):
    """Configure logging."""
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data_from_parquet(parquet_dir: str, years: list) -> pd.DataFrame:
    """Load data from parquet files for specified years."""
    parquet_files = []
    for year in years:
        pattern = os.path.join(parquet_dir, f"matches_{year}.parquet")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No parquet file found for year {year}: {pattern}")
        parquet_files.extend(files)
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    
    dfs = [pd.read_parquet(f) for f in parquet_files]
    combined = pd.concat(dfs, ignore_index=True)
    
    # Ensure tournament_date is datetime
    if 'tournament_date' in combined.columns:
        combined['tournament_date'] = safe_to_datetime(combined['tournament_date'])
    
    return combined


def clean_leakage_columns(df: pd.DataFrame, leak_re) -> pd.DataFrame:
    """Remove columns that contain post-match information."""
    leak_cols = [c for c in df.columns if leak_re.search(c)]
    if leak_cols:
        logging.info(f"Dropping {len(leak_cols)} leakage columns")
    return df.drop(columns=leak_cols, errors='ignore')


def encode_categorical(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Encode categorical variables using config mappings."""
    d = df.copy()
    encodings = config['features']['encodings']
    
    if 'tournament_level' in d.columns:
        d['tournament_level'] = d['tournament_level'].map(encodings['level'])
    if 'tournament_surface' in d.columns:
        d['tournament_surface'] = d['tournament_surface'].map(encodings['surface'])
    if 'round' in d.columns:
        d['round'] = d['round'].map(encodings['round'])
    
    return d


def prepare_data(config: dict, split: str, logger) -> pd.DataFrame:
    """
    Load and prepare data for a specific split.
    
    Args:
        config: Configuration dictionary
        split: 'train', 'calib', or 'test'
        logger: Logger instance
        
    Returns:
        Prepared DataFrame
    """
    years = config['data'][f'{split}_years']
    logger.info(f"Loading {split} data for years: {years}")
    
    # Load from parquet
    parquet_dir = config['data']['parquet_dir']
    df = load_data_from_parquet(parquet_dir, years)
    logger.info(f"Loaded {len(df)} matches")
    
    # Clean leakage
    df = clean_leakage_columns(df, LEAK_RE)
    
    # Encode categorical
    df = encode_categorical(df, config)
    
    # Infer target
    if 'y' not in df.columns:
        if 'Winner' in df.columns:
            winner_val = infer_player1_win_mapping(df)
            df['y'] = (df['Winner'] == winner_val).astype(int)
        else:
            raise ValueError("Cannot determine target variable 'y'")
    
    # Add features
    logger.info("Engineering features...")
    df = add_all_features(df)
    
    # Compute Elo
    logger.info("Computing Elo ratings...")
    df = compute_elo_features(
        df, 
        base=config['features']['elo']['base'], 
        k=config['features']['elo']['k']
    )
    
    return df


def select_features(df: pd.DataFrame, exclude_cols: set) -> list:
    """Select feature columns automatically."""
    exclude = exclude_cols | {'y', 'tournament_date', 'Winner', 'score'}
    features = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, np.int32, np.float32]]
    return features


def create_model(config: dict):
    """Create model based on config."""
    model_type = config['model']['type']
    random_state = config['model']['random_state']
    
    if model_type == 'histgradient':
        params = config['model']['histgradient']
        return HistGradientBoostingModel(
            learning_rate=params['learning_rate'],
            max_iter=params['max_iter'],
            max_depth=params['max_depth'],
            min_samples_leaf=params['min_samples_leaf'],
            l2_regularization=params['l2_regularization'],
            early_stopping=params['early_stopping'],
            validation_fraction=params['validation_fraction'],
            n_iter_no_change=params['n_iter_no_change'],
            random_state=random_state
        )
    elif model_type == 'xgboost':
        params = config['model']['xgboost']
        return XGBoostModel(
            learning_rate=params['learning_rate'],
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_child_weight=params['min_child_weight'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            early_stopping_rounds=params['early_stopping_rounds'],
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train(config: dict, logger):
    """Train model."""
    logger.info("="*60)
    logger.info("TRAINING PHASE")
    logger.info("="*60)
    
    # Load data
    train_df = prepare_data(config, 'train', logger)
    calib_df = prepare_data(config, 'calib', logger)
    
    # Select features
    exclude = set()
    if 'ID_1' in train_df.columns:
        exclude.add('ID_1')
    if 'ID_2' in train_df.columns:
        exclude.add('ID_2')
    
    features = select_features(train_df, exclude)
    logger.info(f"Selected {len(features)} features")
    
    # Prepare matrices
    X_train = train_df[features].fillna(-1).values
    y_train = train_df['y'].values
    X_calib = calib_df[features].fillna(-1).values
    y_calib = calib_df['y'].values
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Calibration set: {len(X_calib)} samples")
    
    # Create and train model
    model = create_model(config)
    
    if config['training']['use_calibration']:
        logger.info("Training with calibration...")
        model.fit_calibrated(
            X_train, y_train,
            X_calib, y_calib,
            method=config['training']['calibration_method']
        )
    else:
        logger.info("Training without calibration...")
        model.fit(X_train, y_train)
    
    # Save model
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    model_path = config['output']['model_path']
    
    bundle = {
        'model': model,
        'features': features,
        'threshold': config['model']['threshold'],
        'config': config,
        'trained_at': datetime.now().isoformat()
    }
    
    joblib.dump(bundle, model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model, features


def evaluate(config: dict, logger, model=None, features=None):
    """Evaluate model on test set."""
    logger.info("="*60)
    logger.info("EVALUATION PHASE")
    logger.info("="*60)
    
    # Load model if not provided
    if model is None or features is None:
        logger.info("Loading trained model...")
        bundle = joblib.load(config['output']['model_path'])
        model = bundle['model']
        features = bundle['features']
    
    # Load test data
    test_df = prepare_data(config, 'test', logger)
    
    # Prepare test matrix
    X_test = test_df[features].fillna(-1).values
    y_test = test_df['y'].values
    
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Get predictions
    proba = model.predict_proba(X_test)[:, 1]
    threshold = config['model']['threshold']
    pred = (proba >= threshold).astype(int)
    
    # Compute metrics
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
    
    acc = accuracy_score(y_test, pred)
    brier = brier_score_loss(y_test, proba)
    logloss = log_loss(y_test, proba)
    alpha = config['training']['alpha']
    objective = alpha * brier + (1 - alpha) * (1 - acc)
    
    results = {
        'n': len(y_test),
        'acc': acc,
        'brier': brier,
        'logloss': logloss,
        'objective': objective
    }
    
    # Print results
    logger.info("\nTest Set Results:")
    logger.info(f"  N:         {results['n']}")
    logger.info(f"  Accuracy:  {results['acc']:.4f}")
    logger.info(f"  Brier:     {results['brier']:.5f}")
    logger.info(f"  LogLoss:   {results['logloss']:.4f}")
    logger.info(f"  Objective: {results['objective']:.5f}")
    
    # Save evaluation report
    report_path = config['output']['eval_report']
    with open(report_path, 'w') as f:
        f.write("Tennis Betting Model - Evaluation Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model Type: {config['model']['type']}\n")
        f.write(f"Evaluated at: {datetime.now().isoformat()}\n\n")
        f.write("Test Set Metrics:\n")
        f.write(f"  Samples:   {results['n']}\n")
        f.write(f"  Accuracy:  {results['acc']:.4f}\n")
        f.write(f"  Brier:     {results['brier']:.5f}\n")
        f.write(f"  LogLoss:   {results['logloss']:.4f}\n")
        f.write(f"  Objective: {results['objective']:.5f}\n")
    
    logger.info(f"Evaluation report saved to {report_path}")
    
    return results


def infer(config: dict, logger, input_file: str = None):
    """Run inference on new data."""
    logger.info("="*60)
    logger.info("INFERENCE PHASE")
    logger.info("="*60)
    
    # Load model
    logger.info("Loading trained model...")
    bundle = joblib.load(config['output']['model_path'])
    model = bundle['model']
    features = bundle['features']
    threshold = bundle['threshold']
    
    # Determine input file
    if input_file is None:
        input_file = config['inference']['input_file']
    
    if input_file is None:
        raise ValueError("No input file specified for inference")
    
    logger.info(f"Loading data from {input_file}")
    
    # Load data (support parquet or CSV)
    if input_file.endswith('.parquet'):
        raw_df = pd.read_parquet(input_file)
    else:
        raw_df = pd.read_csv(input_file, sep=config['data']['csv_sep'])
    
    logger.info(f"Loaded {len(raw_df)} matches")
    
    # Prepare data (similar to training pipeline)
    df = clean_leakage_columns(raw_df, LEAK_RE)
    df = encode_categorical(df, config)
    df = add_all_features(df)
    df = compute_elo_features(
        df, 
        base=config['features']['elo']['base'], 
        k=config['features']['elo']['k']
    )
    
    # Predict
    X = df[features].fillna(-1).values
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    
    # Create clean output with only essential columns and predictions
    result_df = pd.DataFrame()
    
    # Keep only key identifying columns
    keep_cols = ['tournament_date', 'tournament', 'tournament_level', 'tournament_surface',
                 'round', 'ID_1', 'Name_1', 'ID_2', 'Name_2']
    for col in keep_cols:
        if col in raw_df.columns:
            result_df[col] = raw_df[col]
    
    # Add ground truth (transform Winner to match prediction format)
    if 'Winner' in raw_df.columns:
        # Assuming Winner=0 means Player 1 wins, Winner=1 means Player 2 wins
        # Transform to: actual_player1_win = 1 if Player 1 wins, 0 otherwise
        result_df['actual_player1_win'] = (raw_df['Winner'] == 0).astype(int)
    
    # Add predictions
    if config['inference']['include_probabilities']:
        result_df['p_player1_win'] = proba
    result_df['pred_player1_win'] = pred
    
    # Add comparison column if ground truth exists
    if 'actual_player1_win' in result_df.columns:
        result_df['correct'] = (result_df['pred_player1_win'] == result_df['actual_player1_win']).astype(int)
    
    # Save predictions
    os.makedirs(config['inference']['output_dir'], exist_ok=True)
    
    basename = os.path.basename(input_file)
    name_without_ext = os.path.splitext(basename)[0]
    output_path = os.path.join(
        config['inference']['output_dir'], 
        f"predictions_{name_without_ext}.csv"
    )
    
    result_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    # Summary
    n_pred_wins = pred.sum()
    logger.info(f"\nPrediction Summary:")
    logger.info(f"  Total matches: {len(pred)}")
    logger.info(f"  Player 1 wins predicted: {n_pred_wins} ({n_pred_wins/len(pred)*100:.1f}%)")
    logger.info(f"  Average P(Player 1 wins): {proba.mean():.3f}")
    
    return result_df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Tennis Betting Model Runner')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'eval', 'infer', 'all'],
        default='all',
        help='Execution mode'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input file for inference (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger = setup_logging(config['output']['log_level'])
    
    logger.info(f"Loaded configuration from {args.config}")
    logger.info(f"Execution mode: {args.mode}")
    
    model, features = None, None
    
    try:
        if args.mode in ['train', 'all']:
            model, features = train(config, logger)
        
        if args.mode in ['eval', 'all']:
            evaluate(config, logger, model, features)
        
        if args.mode == 'infer':
            infer(config, logger, args.input)
        
        logger.info("\n" + "="*60)
        logger.info("✓ SUCCESS - All operations completed")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\n✗ ERROR: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
