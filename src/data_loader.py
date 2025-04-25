"""
Data loader module for the ML pipeline.
Responsible for loading and initial processing of the dataset.
"""
import pandas as pd
from typing import Tuple, Optional, List


class DataLoader:
    """Loads and performs initial processing of the dataset."""

    def __init__(self, data_path: str):
        """
        Initialize the data loader with the path to the dataset.
        
        Args:
            data_path: Path to the CSV dataset
        """
        self.data_path = data_path
        self.data = None
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from CSV.
        
        Returns:
            Pandas DataFrame containing the dataset
        """
        try:
            self.data = pd.read_csv(self.data_path)
            return self.data
        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")
    
    def split_features_target(
        self, 
        target_col: str, 
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split the dataset into features and target.
        
        Args:
            target_col: Name of the target column
            feature_cols: List of feature column names to include
            
        Returns:
            Tuple of (X, y) where X is the feature DataFrame and y is the target Series
        """
        if self.data is None:
            self.load_data()
        
        if target_col not in self.data.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        if feature_cols:
            missing_cols = [col for col in feature_cols if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Feature columns {missing_cols} not found in dataset")
            X = self.data[feature_cols]
        else:
            # Use all columns except target as features
            X = self.data.drop(columns=[target_col])
        
        y = self.data[target_col]
        
        return X, y
    
    def get_column_types(self) -> dict:
        """
        Get the data types of each column.
        
        Returns:
            Dict mapping column names to their types
        """
        if self.data is None:
            self.load_data()
        
        return {col: str(dtype) for col, dtype in self.data.dtypes.items()}
    
    def get_data_summary(self) -> dict:
        """
        Get a summary of the dataset.
        
        Returns:
            Dict containing dataset summary
        """
        if self.data is None:
            self.load_data()
        
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'column_types': self.get_column_types()
        }