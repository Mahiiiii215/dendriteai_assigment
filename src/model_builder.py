"""
Model builder module for the ML pipeline.
Handles creation and training of machine learning models based on configuration.
"""
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from typing import Dict, List, Any, Tuple, Optional


class ModelBuilder:
    """Builds and trains machine learning models based on configuration."""
    
    def __init__(self, algorithm_config: Dict[str, Any], prediction_type: str):
        """
        Initialize the model builder.
        
        Args:
            algorithm_config: Configuration for the algorithm
            prediction_type: Type of prediction task ('classification' or 'regression')
        """
        self.algorithm_config = algorithm_config
        self.prediction_type = prediction_type
        self.algorithm_name = algorithm_config.get('name', '')
        self.hyperparameters = algorithm_config.get('hyperparameters', {})
    
    def get_base_model(self) -> Tuple[Any, Dict[str, List[Any]]]:
        """
        Get the base model and hyperparameter grid based on configuration.
        
        Returns:
            Tuple of (model, param_grid)
        """
        # Classification models
        if self.prediction_type == 'classification':
            if self.algorithm_name == 'logistic_regression':
                model = LogisticRegression(max_iter=1000, random_state=42)
                param_grid = {
                    'C': self.hyperparameters.get('C', [0.1, 1.0, 10.0]),
                    'penalty': self.hyperparameters.get('penalty', ['l2']),
                    'solver': self.hyperparameters.get('solver', ['liblinear', 'saga'])
                }
            
            elif self.algorithm_name == 'decision_tree':
                model = DecisionTreeClassifier(random_state=42)
                param_grid = {
                    'max_depth': self.hyperparameters.get('max_depth', [None, 5, 10]),
                    'min_samples_split': self.hyperparameters.get('min_samples_split', [2, 5, 10]),
                    'criterion': self.hyperparameters.get('criterion', ['gini', 'entropy'])
                }
            
            elif self.algorithm_name == 'random_forest':
                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'n_estimators': self.hyperparameters.get('n_estimators', [100, 200]),
                    'max_depth': self.hyperparameters.get('max_depth', [None, 5, 10]),
                    'min_samples_split': self.hyperparameters.get('min_samples_split', [2, 5])
                }
            
            elif self.algorithm_name == 'svm':
                model = SVC(probability=True, random_state=42)
                param_grid = {
                    'C': self.hyperparameters.get('C', [0.1, 1.0, 10.0]),
                    'kernel': self.hyperparameters.get('kernel', ['linear', 'rbf']),
                    'gamma': self.hyperparameters.get('gamma', ['scale', 'auto'])
                }
            
            elif self.algorithm_name == 'knn':
                model = KNeighborsClassifier()
                param_grid = {
                    'n_neighbors': self.hyperparameters.get('n_neighbors', [3, 5, 7]),
                    'weights': self.hyperparameters.get('weights', ['uniform', 'distance']),
                    'p': self.hyperparameters.get('p', [1, 2])  # 1 for Manhattan, 2 for Euclidean
                }
            
            else:
                raise ValueError(f"Unsupported classification algorithm: {self.algorithm_name}")
        
        # Regression models
        else:
            if self.algorithm_name == 'linear_regression':
                model = LinearRegression()
                # Linear regression doesn't have hyperparameters to tune
                param_grid = {}
            
            elif self.algorithm_name == 'decision_tree':
                model = DecisionTreeRegressor(random_state=42)
                param_grid = {
                    'max_depth': self.hyperparameters.get('max_depth', [None, 5, 10]),
                    'min_samples_split': self.hyperparameters.get('min_samples_split', [2, 5, 10]),
                    'criterion': self.hyperparameters.get('criterion', ['mse', 'mae'])
                }
            
            elif self.algorithm_name == 'random_forest':
                model = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': self.hyperparameters.get('n_estimators', [100, 200]),
                    'max_depth': self.hyperparameters.get('max_depth', [None, 5, 10]),
                    'min_samples_split': self.hyperparameters.get('min_samples_split', [2, 5])
                }
            
            elif self.algorithm_name == 'svr':
                model = SVR()
                param_grid = {
                    'C': self.hyperparameters.get('C', [0.1, 1.0, 10.0]),
                    'kernel': self.hyperparameters.get('kernel', ['linear', 'rbf']),
                    'gamma': self.hyperparameters.get('gamma', ['scale', 'auto'])
                }
            
            elif self.algorithm_name == 'knn':
                model = KNeighborsRegressor()
                param_grid = {
                    'n_neighbors': self.hyperparameters.get('n_neighbors', [3, 5, 7]),
                    'weights': self.hyperparameters.get('weights', ['uniform', 'distance']),
                    'p': self.hyperparameters.get('p', [1, 2])
                }
            
            else:
                raise ValueError(f"Unsupported regression algorithm: {self.algorithm_name}")
        
        return model, param_grid
    
    def build_model(self, preprocessor: Any, feature_reducer: Optional[Any] = None) -> GridSearchCV:
        """
        Build a complete model pipeline with preprocessing, feature reduction, and model.
        
        Args:
            preprocessor: Preprocessor step for the pipeline
            feature_reducer: Feature reduction step (optional)
            
        Returns:
            GridSearchCV instance with the complete pipeline
        """
        base_model, param_grid = self.get_base_model()
        
        # Build pipeline steps
        steps = [('preprocessor', preprocessor)]
        
        # Add feature reducer if provided
        if feature_reducer is not None:
            steps.append(('feature_reducer', feature_reducer))
        
        # Add the model as the final step
        steps.append(('model', base_model))
        
        # Create pipeline
        pipeline = Pipeline(steps)
        
        # If param_grid is empty, no need for GridSearchCV
        if not param_grid:
            return pipeline
        
        # Add 'model__' prefix to parameter names for GridSearchCV
        prefixed_param_grid = {f'model__{key}': value for key, value in param_grid.items()}
        
        # Create GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            param_grid=prefixed_param_grid,
            cv=5,  # 5-fold cross-validation
            scoring='accuracy' if self.prediction_type == 'classification' else 'neg_mean_squared_error',
            n_jobs=-1
        )
        
        return grid_search