# =========================
# Inference Pipeline
# =========================

import numpy as np
import pandas as pd
import joblib
from typing import Optional

from config import ELO_BASE, ELO_K
from utils import encode_categorical, clean_leakage_columns
from features import add_all_features, compute_elo_features


def prepare_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare raw data for inference.
    
    Applies all necessary transformations used during training:
    - Encoding categorical variables
    - Removing leakage columns
    - Engineering features
    - Computing Elo ratings
    
    Args:
        df: Raw input DataFrame
        
    Returns:
        Prepared DataFrame ready for prediction
    """
    d = df.copy()
    
    # Clean and encode
    d = clean_leakage_columns(d)
    d = encode_categorical(d)
    
    # Ensure dates
    if "tournament_date" in d.columns:
        d["tournament_date"] = pd.to_datetime(d["tournament_date"], errors="coerce")
    
    # Engineer features
    d = add_all_features(d)
    
    # Compute Elo (requires historical ordering)
    # For pure future matches, you need Elo state at that time
    # Here we compute within the provided df for batch scoring
    if all(c in d.columns for c in ["ID_1", "ID_2", "tournament_date"]):
        if "y" not in d.columns:
            d["y"] = np.nan
        d = compute_elo_features(d, base=ELO_BASE, k=ELO_K)
    
    return d


def load_model(model_path: str) -> dict:
    """
    Load trained model bundle.
    
    Args:
        model_path: Path to saved model (.joblib file)
        
    Returns:
        Dictionary containing model, features, and metadata
    """
    bundle = joblib.load(model_path)
    
    required_keys = ["model", "features", "threshold"]
    for key in required_keys:
        if key not in bundle:
            raise ValueError(f"Model bundle missing required key: {key}")
    
    return bundle


def predict_from_dataframe(
    model_bundle: dict,
    df: pd.DataFrame,
    return_probabilities: bool = True
) -> pd.DataFrame:
    """
    Make predictions on a prepared DataFrame.
    
    Args:
        model_bundle: Dictionary from load_model()
        df: Prepared DataFrame (after prepare_for_inference)
        return_probabilities: Whether to include probability column
        
    Returns:
        DataFrame with predictions added
    """
    model = model_bundle["model"]
    features = model_bundle["features"]
    threshold = model_bundle.get("threshold", 0.5)
    
    # Check for missing features
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    
    # Prepare features (fill missing with -1)
    X = df[features].fillna(-1).values
    
    # Predict
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    
    # Create output with only essential columns and predictions
    result = pd.DataFrame()
    
    # Keep only key identifying columns if they exist
    keep_cols = ['tournament_date', 'tournament', 'tournament_level', 'tournament_surface', 
                 'round', 'ID_1', 'Name_1', 'ID_2', 'Name_2']
    for col in keep_cols:
        if col in df.columns:
            result[col] = df[col]
    
    # Add ground truth (transform Winner to match prediction format)
    if 'Winner' in df.columns:
        # Assuming Winner=0 means Player 1 wins, Winner=1 means Player 2 wins
        # Transform to: actual_player1_win = 1 if Player 1 wins, 0 otherwise
        result['actual_player1_win'] = (df['Winner'] == 0).astype(int)
    
    # Add predictions
    if return_probabilities:
        result["p_player1_win"] = proba
    result["pred_player1_win"] = pred
    
    # Add comparison column if ground truth exists
    if 'actual_player1_win' in result.columns:
        result['correct'] = (result['pred_player1_win'] == result['actual_player1_win']).astype(int)
    
    return result


def predict_from_csv(
    model_path: str,
    csv_path: str,
    output_path: Optional[str] = None,
    sep: str = ";"
) -> pd.DataFrame:
    """
    Complete inference pipeline from CSV file.
    
    Args:
        model_path: Path to saved model
        csv_path: Path to input CSV file
        output_path: Optional path to save predictions
        sep: CSV separator
        
    Returns:
        DataFrame with predictions
    """
    # Load model
    print(f"[INFO] Loading model from {model_path}")
    bundle = load_model(model_path)
    
    # Load and prepare data
    print(f"[INFO] Loading data from {csv_path}")
    raw = pd.read_csv(csv_path, sep=sep)
    print(f"[INFO] Loaded {len(raw)} matches")
    
    # Prepare for inference
    print("[INFO] Preparing features...")
    prepared = prepare_for_inference(raw)
    
    # Predict
    print("[INFO] Making predictions...")
    result = predict_from_dataframe(bundle, prepared)
    
    # Save if requested
    if output_path:
        result.to_csv(output_path, index=False)
        print(f"[SAVE] Predictions saved to {output_path}")
    
    # Summary statistics
    n_player1_wins = result["pred_player1_win"].sum()
    avg_prob = result["p_player1_win"].mean()
    
    print(f"\n[SUMMARY]")
    print(f"  Total predictions: {len(result)}")
    print(f"  Player 1 predicted wins: {n_player1_wins} ({n_player1_wins/len(result)*100:.1f}%)")
    print(f"  Average P(Player 1 wins): {avg_prob:.3f}")
    
    if 'correct' in result.columns:
        accuracy = result['correct'].mean()
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return result


def batch_predict(
    model_path: str,
    input_files: list,
    output_dir: str,
    sep: str = ";"
) -> dict:
    """
    Run predictions on multiple files.
    
    Args:
        model_path: Path to saved model
        input_files: List of input CSV paths
        output_dir: Directory to save predictions
        sep: CSV separator
        
    Returns:
        Dictionary mapping input file to prediction DataFrame
    """
    import os
    
    bundle = load_model(model_path)
    results = {}
    
    os.makedirs(output_dir, exist_ok=True)
    
    for input_file in input_files:
        basename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, f"pred_{basename}")
        
        print(f"\n[BATCH] Processing {basename}...")
        result = predict_from_csv(model_path, input_file, output_file, sep)
        results[input_file] = result
    
    return results


# Example usage
if __name__ == "__main__":
    # Example: predict on a single file
    model_path = "./results/tennis_model.joblib"
    input_csv = "/home/tourmii/Documents/Projects/tennis-betting/data_tennis_match_reduced/backtest/matches_with_odds_2024.csv"
    output_csv = "./results/predictions_2024.csv"
    
    predictions = predict_from_csv(
        model_path=model_path,
        csv_path=input_csv,
        output_path=output_csv,
        sep=";"
    )
    
    print("\n[DONE] Inference complete!")
    print(f"Predictions shape: {predictions.shape}")
    print(f"\nFirst few predictions:")
    print(predictions[["p_player1_win", "pred_player1_win"]].head())