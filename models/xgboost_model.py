# =========================
# XGBoost Model
# =========================


import xgboost as xgb
from base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost model for tennis prediction.
    
    Optimized for Brier score (calibration) while maintaining accuracy.
    """
    
    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        n_estimators: int = 100,
        min_child_weight: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0.1,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42
    ):
        """
        Initialize XGBoost model.
        
        Args:
            max_depth: Maximum depth of trees
            learning_rate: Learning rate (eta)
            n_estimators: Number of boosting rounds
            min_child_weight: Minimum sum of instance weight in child
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            gamma: Minimum loss reduction for split
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            random_state: Random seed
        """
        
        super().__init__(random_state)
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
    
    def _create_model(self):
        """Create XGBClassifier with configured parameters."""
        return xgb.XGBClassifier(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )


# Preset configurations
def create_default_xgboost() -> XGBoostModel:
    """Create model with default parameters (optimized for Brier)."""
    
    return XGBoostModel(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=100,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        random_state=42
    )


def create_fast_xgboost() -> XGBoostModel:
    """Create faster model with fewer estimators."""
    
    return XGBoostModel(
        max_depth=4,
        learning_rate=0.1,
        n_estimators=50,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )


def create_deep_xgboost() -> XGBoostModel:
    """Create deeper, more complex model."""
    return XGBoostModel(
        max_depth=8,
        learning_rate=0.03,
        n_estimators=150,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        reg_lambda=2.0,
        random_state=42
    )