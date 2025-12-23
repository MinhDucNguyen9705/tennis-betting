# =========================
# Base Model Interface
# =========================

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


class BaseModel(ABC):
    """Abstract base class for tennis prediction models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.calibrated_model = None
        self.calibrator = None  # For manual calibration
        
    @abstractmethod
    def _create_model(self):
        """Create and return the base model."""
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the base model."""
        self.model = self._create_model()
        self.model.fit(X, y)
        return self
    
    def fit_calibrated(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        method: str = "isotonic"
    ):
        """
        Fit model with calibration.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_calib: Calibration features
            y_calib: Calibration labels
            method: Calibration method ('isotonic' or 'sigmoid')
            
        Returns:
            self
        """
        # Fit base model on training data
        self.fit(X_train, y_train)
        
        # Calibrate on calibration set using the newer API
        from sklearn.calibration import calibration_curve
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        
        # Get predictions on calibration set
        proba_calib = self.model.predict_proba(X_calib)[:, 1]
        
        if method == "isotonic":
            # Use isotonic regression for calibration
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(proba_calib, y_calib)
            self.calibrated_model = self  # Use self with calibrator
        elif method == "sigmoid":
            # Use Platt scaling (logistic regression)
            self.calibrator = LogisticRegression()
            self.calibrator.fit(proba_calib.reshape(-1, 1), y_calib)
            self.calibrated_model = self
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (uses calibrated model if available)."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() or fit_calibrated() first.")
        
        # Get base predictions
        proba = self.model.predict_proba(X)
        
        # Apply calibration if available
        if self.calibrator is not None:
            proba_class1 = proba[:, 1]
            if hasattr(self.calibrator, 'predict'):
                # Isotonic regression or similar
                calibrated_proba = self.calibrator.predict(proba_class1)
            else:
                # Logistic regression (sigmoid)
                calibrated_proba = self.calibrator.predict_proba(proba_class1.reshape(-1, 1))[:, 1]
            
            # Reconstruct probability array
            proba = np.column_stack([1 - calibrated_proba, calibrated_proba])
        
        return proba
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)
    
    def evaluate(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        threshold: float = 0.5,
        ids: np.ndarray = None
    ) -> dict:
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y: True labels
            threshold: Classification threshold
            ids: Match IDs for paired bootstrap
            
        Returns:
            Dictionary with metrics
        """
        proba = self.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)
        
        acc = accuracy_score(y, pred)
        brier = brier_score_loss(y, proba)
        ll = log_loss(y, proba)
        
        result = {
            "acc": acc,
            "brier": brier,
            "logloss": ll,
            "p": proba,
            "y": y,
            "n": len(y)
        }
        
        if ids is not None:
            result["ids"] = ids
            
        return result


def evaluate_on_dataframe(
    model: BaseModel,
    df: pd.DataFrame,
    feature_cols: list,
    key_col: str = None,
    threshold: float = 0.5
) -> dict:
    """
    Evaluate model on a DataFrame.
    
    Args:
        model: Fitted BaseModel instance
        df: DataFrame with features and labels
        feature_cols: List of feature column names
        key_col: Column name for match IDs
        threshold: Classification threshold
        
    Returns:
        Dictionary with evaluation metrics
    """
    df_eval = df.dropna(subset=feature_cols + ["y"]).copy()
    
    if key_col and key_col not in df_eval.columns:
        raise ValueError(f"key_col={key_col} not found in dataframe.")
    
    X = df_eval[feature_cols].values
    y = df_eval["y"].astype(int).values
    ids = df_eval[key_col].astype(str).values if key_col else None
    
    return model.evaluate(X, y, threshold=threshold, ids=ids)