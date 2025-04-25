"""
Preprocessing module for the ML pipeline.
Handles feature preprocessing including imputation of missing values.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from typing import Dict, List, Any


class CustomImputer(BaseEstimator, TransformerMixin):
    """
    Custom imputer that applies different imputation methods to different columns.
    Implements scikit-learn's transformer interface.
    """
    
    def __init__(self, imputation_methods: Dict[str, str]):
        """
        Initialize the imputer with a mapping of columns to imputation methods.
        
        Args:
            imputation_methods: Dict mapping column names to imputation methods
                                ('mean', 'median', 'mode', 'constant', etc.)
        """
        self.imputation_methods = imputation_methods
        self.imputers = {}
        self.feature_names = None
    
    def _get_imputer(self, method: str) -> SimpleImputer:
        """
        Get the appropriate scikit-learn imputer for a given method.
        
        Args:
            method: Imputation method ('mean', 'median', 'most_frequent', 'constant')
            
        Returns:
            SimpleImputer instance
        """
        if method == 'mean':
            return SimpleImputer(strategy='mean')
        elif method == 'median':
            return SimpleImputer(strategy='median')
        elif method == 'mode':
            return SimpleImputer(strategy='most_frequent')
        elif method.startswith('constant:'):
            _, value = method.split(':', 1)
            return SimpleImputer(strategy='constant', fill_value=value)
        else:
            # Default to mean imputation
            return SimpleImputer(strategy='mean')
    
    def fit(self, X: pd.DataFrame, y=None) -> 'CustomImputer':
        """
        Fit the imputer on the training data.
        
        Args:
            X: Training data
            y: Target values (unused)
            
        Returns:
            Self
        """
        self.feature_names = X.columns.tolist()
        
        # For each column that needs imputation, fit the appropriate imputer
        for col, method in self.imputation_methods.items():
            if col in X.columns:
                # Create a new imputer for this column
                self.imputers[col] = self._get_imputer(method)
                # Fit on the column data
                self.imputers[col].fit(X[[col]])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by applying the fitted imputers.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data with imputed values
        """
        X_transformed = X.copy()
        
        # Apply imputation to each column
        for col, imputer in self.imputers.items():
            if col in X.columns:
                X_transformed[col] = imputer.transform(X[[col]])
        
        return X_transformed


class Preprocessor:
    """Handles preprocessing of data for the ML pipeline."""
    
    def __init__(self, imputation_methods: Dict[str, str]):
        """
        Initialize the preprocessor with imputation methods.
        
        Args:
            imputation_methods: Dict mapping column names to imputation methods
        """
        self.imputation_methods = imputation_methods
        self.imputer = CustomImputer(imputation_methods)
    
    def get_preprocessor(self) -> CustomImputer:
        """
        Get the preprocessor transformer.
        
        Returns:
            CustomImputer instance
        """
        return self.imputer