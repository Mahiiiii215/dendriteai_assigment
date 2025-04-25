"""
Pipeline orchestrator module for the ML pipeline.
Coordinates the entire ML pipeline execution.
"""
import logging
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import train_test_split
import time
import json

from .config_parser import ConfigParser
from .data_loader import DataLoader
from .preprocessing import Preprocessor
from .feature_reduction import FeatureReducer
from .model_builder import ModelBuilder
from .evaluation import ModelEvaluator


class Pipeline:
    """
    Main pipeline orchestrator that coordinates the ML pipeline execution.
    """
    
    def __init__(self, config_path: str, data_path: str):
        """
        Initialize the pipeline with paths to configuration and data.
        
        Args:
            config_path: Path to JSON configuration file
            data_path: Path to CSV dataset file
        """
        # Setup logging
        self._setup_logging()
        
        self.config_path = config_path
        self.data_path = data_path
        
        # Load configuration
        self.logger.info(f"Loading configuration from {config_path}")
        self.config_parser = ConfigParser(config_path)
        
        # Initialize data loader
        self.logger.info(f"Initializing data loader for {data_path}")
        self.data_loader = DataLoader(data_path)
        
        # Set prediction type
        self.prediction_type = self.config_parser.get_prediction_type()
        self.logger.info(f"Prediction type: {self.prediction_type}")
        
        # Results storage
        self.results = {}
    
    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('ml_pipeline.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run(self) -> Dict[str, Any]:
        """
        Run the full ML pipeline.
        
        Returns:
            Dict containing pipeline results
        """
        start_time = time.time()
        self.logger.info("Starting ML pipeline execution")
        
        try:
            # Load and prepare data
            X, y = self._prepare_data()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            self.logger.info(f"Data split into training ({X_train.shape}) and test ({X_test.shape})")
            
            # Set up preprocessor
            preprocessor = self._setup_preprocessor()
            
            # Set up feature reducer
            feature_reducer = self._setup_feature_reducer()
            
            # Get selected algorithms
            selected_algorithms = self.config_parser.get_selected_algorithms()
            self.logger.info(f"Selected {len(selected_algorithms)} algorithms for training")
            
            # Train and evaluate each selected algorithm
            for algo_config in selected_algorithms:
                algo_name = algo_config.get('name', 'unknown')
                self.logger.info(f"Training model: {algo_name}")
                
                # Build model
                model_builder = ModelBuilder(algo_config, self.prediction_type)
                model = model_builder.build_model(preprocessor, feature_reducer)
                
                # Train model
                self.logger.info(f"Fitting model {algo_name}")
                model.fit(X_train, y_train)
                
                # Evaluate model
                self.logger.info(f"Evaluating model {algo_name}")
                evaluator = ModelEvaluator(self.prediction_type)
                metrics = evaluator.evaluate(model, X_test, y_test, X_train, y_train)
                
                # Store results
                self.results[algo_name] = {
                    'metrics': metrics,
                    'formatted_results': evaluator.format_results(metrics)
                }
                
                # Log results
                self.logger.info(f"Results for {algo_name}:\n{evaluator.format_results(metrics)}")
            
            # Calculate execution time
            execution_time = time.time() - start_time
            self.logger.info(f"Pipeline execution completed in {execution_time:.2f} seconds")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error in pipeline execution: {e}", exc_info=True)
            raise
    
    def _prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare the dataset.
        
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        # Load data
        self.data_loader.load_data()
        
        # Get feature and target configurations
        target = self.config_parser.get_target_variable()
        feature_names = self.config_parser.get_feature_names()
        
        self.logger.info(f"Target variable: {target}")
        self.logger.info(f"Features: {feature_names}")
        
        # Split features and target
        X, y = self.data_loader.split_features_target(target, feature_names)
        
        return X, y
    
    def _setup_preprocessor(self) -> Any:
        """
        Set up the data preprocessor.
        
        Returns:
            Preprocessor transformer
        """
        # Get imputation methods
        imputation_methods = self.config_parser.get_imputation_methods()
        self.logger.info(f"Imputation methods: {imputation_methods}")
        
        # Create preprocessor
        preprocessor = Preprocessor(imputation_methods)
        return preprocessor.get_preprocessor()
    
    def _setup_feature_reducer(self) -> Optional[Any]:
        """
        Set up the feature reducer.
        
        Returns:
            Feature reducer transformer or None
        """
        # Get feature reduction config
        feature_reduction_config = self.config_parser.get_feature_reduction()
        method = feature_reduction_config.get('method', 'none')
        
        self.logger.info(f"Feature reduction method: {method}")
        
        if method == 'none':
            return None
        
        # Create feature reducer
        feature_reducer = FeatureReducer(feature_reduction_config, self.prediction_type)
        return feature_reducer.get_reducer()
    
    def save_results(self, output_path: str):
        """
        Save pipeline results to a JSON file.
        
        Args:
            output_path: Path to save results
        """
        # Convert results to a serializable format
        serializable_results = {}
        for algo_name, result in self.results.items():
            metrics = result['metrics'].copy()
            
            # Remove non-serializable objects
            if 'confusion_matrix' in metrics:
                metrics['confusion_matrix'] = metrics['confusion_matrix']
            
            # Convert numpy values to Python types
            for key, value in metrics.items():
                if hasattr(value, 'tolist'):
                    metrics[key] = value.tolist()
                elif hasattr(value, 'item'):
                    metrics[key] = value.item()
            
            serializable_results[algo_name] = {
                'metrics': metrics,
                'formatted_results': result['formatted_results']
            }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")