"""
Configuration parser module for the ML pipeline.
Responsible for loading and parsing the JSON configuration file.
"""
import json
from typing import Dict, Any, List, Optional


class ConfigParser:
    """Parses the JSON configuration file for ML pipeline settings."""

    def __init__(self, config_path: str):
        """
        Initialize the config parser with the path to the configuration file.
        
        Args:
            config_path: Path to the JSON configuration file
        """
        self.config_path = config_path
        self.config_data = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load the JSON configuration file.
        
        Returns:
            Dict containing the configuration data
        """
        try:
            with open(self.config_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            raise ValueError(f"Error loading configuration file: {e}")
    
    def get_target_variable(self) -> str:
        """
        Get the target variable from the configuration.
        
        Returns:
            Name of the target variable
        """
        try:
            return self.config_data.get('target_variable', '')
        except KeyError:
            raise ValueError("Target variable not found in configuration")
    
    def get_features(self) -> List[Dict[str, Any]]:
        """
        Get the features configuration.
        
        Returns:
            List of feature configurations
        """
        try:
            return self.config_data.get('features', [])
        except KeyError:
            raise ValueError("Features not found in configuration")
    
    def get_feature_names(self) -> List[str]:
        """
        Get a list of feature names.
        
        Returns:
            List of feature names
        """
        features = self.get_features()
        return [feature.get('name') for feature in features]
    
    def get_imputation_methods(self) -> Dict[str, str]:
        """
        Get the imputation methods for each feature.
        
        Returns:
            Dict mapping feature names to their imputation methods
        """
        features = self.get_features()
        return {feature.get('name'): feature.get('imputation_method', 'mean') 
                for feature in features}
    
    def get_feature_reduction(self) -> Dict[str, Any]:
        """
        Get feature reduction configuration.
        
        Returns:
            Feature reduction configuration
        """
        try:
            return self.config_data.get('feature_reduction', {'method': 'none'})
        except KeyError:
            return {'method': 'none'}
    
    def get_algorithms(self) -> List[Dict[str, Any]]:
        """
        Get the configured algorithms.
        
        Returns:
            List of algorithm configurations
        """
        try:
            return self.config_data.get('algorithms', [])
        except KeyError:
            raise ValueError("Algorithms not found in configuration")
    
    def get_selected_algorithms(self) -> List[Dict[str, Any]]:
        """
        Get only the selected algorithms.
        
        Returns:
            List of selected algorithm configurations
        """
        algorithms = self.get_algorithms()
        return [algo for algo in algorithms if algo.get('is_selected', False)]
    
    def get_prediction_type(self) -> str:
        """
        Get the prediction type (classification or regression).
        
        Returns:
            Prediction type string
        """
        try:
            return self.config_data.get('prediction_type', 'classification')
        except KeyError:
            return 'classification'