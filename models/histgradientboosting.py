# =========================
# HistGradientBoosting Model
# =========================

from sklearn.ensemble import HistGradientBoostingClassifier
from base_model import BaseModel


class HistGradientBoostingModel(BaseModel):
    """
    HistGradientBoosting model for tennis prediction.
    
    Optimized for Brier score (calibration) while maintaining accuracy.
    """
    
    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 50,
        max_iter: int = 100,
        l2_regularization: float = 1.0,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        random_state: int = 42
    ):
        """
        Initialize HistGradientBoosting model.
        
        Args:
            max_depth: Maximum depth of trees
            learning_rate: Learning rate (shrinkage)
            max_leaf_nodes: Maximum number of leaf nodes
            min_samples_leaf: Minimum samples per leaf (helps calibration)
            max_iter: Number of boosting iterations
            l2_regularization: L2 regularization parameter
            early_stopping: Whether to use early stopping
            validation_fraction: Fraction of data for validation
            n_iter_no_change: Iterations without improvement before stopping
            random_state: Random seed
        """
        super().__init__(random_state)
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.max_iter = max_iter
        self.l2_regularization = l2_regularization
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
    
    def _create_model(self):
        """Create HistGradientBoostingClassifier with configured parameters."""
        return HistGradientBoostingClassifier(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            max_iter=self.max_iter,
            l2_regularization=self.l2_regularization,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state
        )


# Preset configurations
def create_default_histgradient() -> HistGradientBoostingModel:
    """Create model with default parameters (optimized for Brier)."""
    return HistGradientBoostingModel(
        max_depth=6,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        max_iter=100,
        random_state=42
    )


def create_fast_histgradient() -> HistGradientBoostingModel:
    """Create faster model with fewer iterations."""
    return HistGradientBoostingModel(
        max_depth=4,
        learning_rate=0.1,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        max_iter=50,
        random_state=42
    )


def create_deep_histgradient() -> HistGradientBoostingModel:
    """Create deeper, more complex model."""
    return HistGradientBoostingModel(
        max_depth=8,
        learning_rate=0.03,
        max_leaf_nodes=63,
        min_samples_leaf=30,
        max_iter=150,
        random_state=42
    )