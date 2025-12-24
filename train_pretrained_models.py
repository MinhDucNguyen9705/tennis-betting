#!/usr/bin/env python3
"""
Train pre-trained models for tennis betting inference.

Trains 5 model types on 2021-2023 data and saves them as joblib files
for direct inference without retraining.

Usage:
    python train_pretrained_models.py
"""

import os
import sys
import joblib
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.backtest_pipeline import train_model_only, MODEL_TYPES

# Configuration
TRAINING_YEARS = [2021, 2022, 2023]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained_models')

# All model types to train
MODEL_TYPES_TO_TRAIN = [
    'svm',
    'catboost', 
    'hist_gradient_boosting',
    'logistic_regression',
    'random_forest',
]


def get_model_filename(model_type: str) -> str:
    """Generate standardized filename for pre-trained model."""
    year_range = f"{min(TRAINING_YEARS)}_{max(TRAINING_YEARS)}"
    return f"{model_type}_{year_range}.joblib"


def train_and_save_model(model_type: str) -> bool:
    """Train a single model and save it as joblib."""
    print(f"\n{'='*60}")
    print(f"Training: {MODEL_TYPES.get(model_type, model_type)}")
    print(f"Years: {TRAINING_YEARS}")
    print('='*60)
    
    try:
        # CatBoost doesn't work well with sklearn's CalibratedClassifierCV
        # due to missing __sklearn_tags__ attribute
        use_calibration = model_type != 'catboost'
        
        # Train model
        model_bundle = train_model_only(
            train_years=TRAINING_YEARS,
            use_calibration=use_calibration,
            model_type=model_type
        )
        
        # Add metadata
        model_bundle['pretrained'] = True
        model_bundle['pretrained_years'] = TRAINING_YEARS
        model_bundle['saved_at'] = datetime.now().isoformat()
        
        # Save to file
        filename = get_model_filename(model_type)
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        joblib.dump(model_bundle, filepath)
        
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"✅ Saved: {filename} ({file_size:.2f} MB)")
        print(f"   Features: {len(model_bundle.get('features', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to train {model_type}: {str(e)}")
        return False


def main():
    """Train all models and save as joblib files."""
    print("\n" + "="*60)
    print("PRE-TRAINED MODELS GENERATOR")
    print(f"Training Years: {TRAINING_YEARS}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("="*60)
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results = {}
    
    for model_type in MODEL_TYPES_TO_TRAIN:
        success = train_and_save_model(model_type)
        results[model_type] = success
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    success_count = sum(1 for v in results.values() if v)
    total = len(results)
    
    for model_type, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {MODEL_TYPES.get(model_type, model_type):25s} {status}")
    
    print(f"\nTotal: {success_count}/{total} models trained successfully")
    print(f"Models saved to: {OUTPUT_DIR}/")
    
    # List saved files
    print("\nSaved files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.joblib'):
            size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / (1024*1024)
            print(f"  - {f} ({size:.2f} MB)")


if __name__ == "__main__":
    main()
