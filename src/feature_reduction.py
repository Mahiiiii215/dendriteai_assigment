"""
Feature reduction module for the ML pipeline.
Implements various feature selection and dimensionality reduction techniques.
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, List, Any, Optional, Union


class FeatureReducer:
    """Handles feature reduction for the ML pipeline."""
    
    def __init__(self, config: Dict[str, Any], prediction_type: str = 'classification'):
        """
        Initialize the feature reducer with a configuration.
        
        Args:
            config: Feature reduction configuration
            prediction_type: Type of prediction task ('classification' or 'regression')
        """
        self.config = config
        self.prediction_type = prediction_type
        self.method = config.get('method', 'none')
        self.n_components = config.get('n_components', None)
        self.threshold = config.get('threshold', 0.1)
    
    def get_reducer(self) -> Optional[BaseEstimator]:
        """
        Get the appropriate feature reducer based on configuration.
        
        Returns:
            Scikit-learn transformer for feature reduction or None if no reduction
        """
        if self.method == 'none':
            return None
        
        if self.method == 'correlation':
            # Use SelectKBest with appropriate scoring function
            if self.prediction_type == 'classification':
                score_func = f_classif
            else:
                score_func = f_regression
            
            # If n_components is specified, use it; otherwise, use all features
            if self.n_components:
                return SelectKBest(score_func=score_func, k=self.n_components)
            else:
                return SelectKBest(score_func=score_func, k='all')
        
        elif self.method == 'tree_based':
            # Use tree-based feature selection
            if self.prediction_type == 'classification':
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            
            return SelectFromModel(
                estimator=estimator, 
                threshold=self.threshold,
                prefit=False
            )
        
        elif self.method == 'pca':
            # Use PCA for dimensionality reduction
            n_components = self.n_components if self.n_components else 'mle'
            return PCA(n_components=n_components, random_state=42)
        
        else:
            raise ValueError(f"Unknown feature reduction method: {self.method}")


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer for selecting features based on correlation with target.
    Useful when you need more control than SelectKBest provides.
    """
    
    def __init__(self, threshold: float = 0.1, top_k: Optional[int] = None):
        """
        Initialize the correlation selector.
        
        Args:
            threshold: Minimum absolute correlation threshold
            top_k: Number of top features to select (if specified)
        """
        self.threshold = threshold
        self.top_k = top_k
        self.selected_features_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'CorrelationSelector':
        """
        Fit the selector by calculating correlations with the target.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Self
        """
        # Calculate correlation of each feature with the target
        corr = {}
        for col in X.columns:
            corr[col] = abs(pd.Series(X[col]).corr(y))
        
        # Filter features based on threshold
        selected = {k: v for k, v in corr.items() if v >= self.threshold}
        
        # If top_k is specified, select top k features
        if self.top_k and len(selected) > self.top_k:
            selected = dict(sorted(selected.items(), key=lambda x: x[1], reverse=True)[:self.top_k])
        
        self.selected_features_ = list(selected.keys())
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform by selecting only the chosen features.
        
        Args:
            X: Feature data
            
        Returns:
            DataFrame with only selected features
        """
        if self.selected_features_ is None:
            raise ValueError("Transformer not fitted. Call fit before transform.")
        
        return X[self.selected_features_]