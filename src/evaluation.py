"""
Evaluation module for the ML pipeline.
Handles model evaluation metrics calculation and reporting.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from typing import Dict, Any, List, Tuple, Optional
import logging


class ModelEvaluator:
    """Evaluates model performance using appropriate metrics."""
    
    def __init__(self, prediction_type: str):
        """
        Initialize the evaluator.
        
        Args:
            prediction_type: Type of prediction ('classification' or 'regression')
        """
        self.prediction_type = prediction_type
        self.logger = logging.getLogger(__name__)
    
    def evaluate(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            model: Trained model
            X: Test feature data
            y: Test target data
            X_train: Training feature data (optional, for additional metrics)
            y_train: Training target data (optional, for additional metrics)
            
        Returns:
            Dict of evaluation metrics
        """
        if self.prediction_type == 'classification':
            return self._evaluate_classification(model, X, y, X_train, y_train)
        else:
            return self._evaluate_regression(model, X, y, X_train, y_train)
    
    def _evaluate_classification(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a classification model.
        
        Args:
            model: Trained model
            X: Test feature data
            y: Test target data
            X_train: Training feature data (optional)
            y_train: Training target data (optional)
            
        Returns:
            Dict of classification metrics
        """
        # Make predictions
        y_pred = model.predict(X)
        
        # Get probabilities if the model supports it
        try:
            y_proba = model.predict_proba(X)
        except (AttributeError, NotImplementedError):
            y_proba = None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted'),
        }
        
        # Add confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Add classification report
        metrics['classification_report'] = classification_report(y, y_pred)
        
        # Calculate training metrics if training data is provided
        if X_train is not None and y_train is not None:
            y_train_pred = model.predict(X_train)
            metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
            metrics['train_f1'] = f1_score(y_train, y_train_pred, average='weighted')
        
        # Add best parameters if it's a GridSearchCV model
        if hasattr(model, 'best_params_'):
            metrics['best_params'] = model.best_params_
        
        self.logger.info(f"Classification metrics: {metrics}")
        return metrics
    
    def _evaluate_regression(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a regression model.
        
        Args:
            model: Trained model
            X: Test feature data
            y: Test target data
            X_train: Training feature data (optional)
            y_train: Training target data (optional)
            
        Returns:
            Dict of regression metrics
        """
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        
        # Calculate training metrics if training data is provided
        if X_train is not None and y_train is not None:
            y_train_pred = model.predict(X_train)
            metrics['train_mse'] = mean_squared_error(y_train, y_train_pred)
            metrics['train_r2'] = r2_score(y_train, y_train_pred)
        
        # Add best parameters if it's a GridSearchCV model
        if hasattr(model, 'best_params_'):
            metrics['best_params'] = model.best_params_
        
        self.logger.info(f"Regression metrics: {metrics}")
        return metrics
    
    def format_results(self, metrics: Dict[str, Any]) -> str:
        """
        Format evaluation results for display.
        
        Args:
            metrics: Dict of evaluation metrics
            
        Returns:
            Formatted string of results
        """
        if self.prediction_type == 'classification':
            return self._format_classification_results(metrics)
        else:
            return self._format_regression_results(metrics)
    
    def _format_classification_results(self, metrics: Dict[str, Any]) -> str:
        """Format classification results."""
        result = "Classification Model Evaluation:\n"
        result += f"Accuracy: {metrics['accuracy']:.4f}\n"
        result += f"Precision: {metrics['precision']:.4f}\n"
        result += f"Recall: {metrics['recall']:.4f}\n"
        result += f"F1 Score: {metrics['f1']:.4f}\n"
        
        if 'train_accuracy' in metrics:
            result += f"\nTraining Accuracy: {metrics['train_accuracy']:.4f}\n"
            result += f"Training F1 Score: {metrics['train_f1']:.4f}\n"
        
        if 'best_params' in metrics:
            result += "\nBest Parameters:\n"
            for param, value in metrics['best_params'].items():
                result += f"  {param}: {value}\n"
        
        return result
    
    def _format_regression_results(self, metrics: Dict[str, Any]) -> str:
        """Format regression results."""
        result = "Regression Model Evaluation:\n"
        result += f"Mean Squared Error (MSE): {metrics['mse']:.4f}\n"
        result += f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}\n"
        result += f"Mean Absolute Error (MAE): {metrics['mae']:.4f}\n"
        result += f"R² Score: {metrics['r2']:.4f}\n"
        
        if 'train_mse' in metrics:
            result += f"\nTraining MSE: {metrics['train_mse']:.4f}\n"
            result += f"Training R² Score: {metrics['train_r2']:.4f}\n"
        
        if 'best_params' in metrics:
            result += "\nBest Parameters:\n"
            for param, value in metrics['best_params'].items():
                result += f"  {param}: {value}\n"
        
        return result