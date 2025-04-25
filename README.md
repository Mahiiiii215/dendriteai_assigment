# Machine Learning Pipeline

A flexible, configuration-driven machine learning pipeline that dynamically executes ML tasks based on a JSON configuration file.

## Overview

This project implements a generic machine learning pipeline that:

1. Parses a JSON configuration file (`algoparams_from_ui.json`)
2. Loads and processes a dataset
3. Applies specified preprocessing steps
4. Performs feature reduction if configured
5. Builds and trains specified ML models
6. Evaluates model performance
7. Outputs comprehensive results

The pipeline is designed to be flexible and can handle different datasets, algorithms, and configurations with minimal changes to the code.

## Features

- JSON-driven pipeline configuration
- Support for multiple imputation methods per feature
- Feature reduction options:
  - No reduction
  - Correlation with target
  - Tree-based feature selection
  - PCA
- Support for multiple ML algorithms
- Hyperparameter tuning with grid search
- Comprehensive model evaluation
- Modular architecture for easy extension

## Project Structure

```
/
├── main.py             # Main entry point
├── src/                # Source code directory
│   ├── __init__.py
│   ├── config_parser.py     # JSON configuration parser
│   ├── data_loader.py       # Dataset loading and preparation
│   ├── preprocessing.py     # Data preprocessing and imputation
│   ├── feature_reduction.py # Feature selection/reduction
│   ├── model_builder.py     # ML model construction
│   ├── evaluation.py        # Model evaluation
│   └── pipeline.py          # Pipeline orchestrator
├── iris.csv            # Sample dataset
├── algoparams_from_ui.json  # Sample configuration
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Run with default configuration and dataset
python main.py

# Run with custom configuration and dataset
python main.py --config my_config.json --data my_dataset.csv --output my_results.json
```

## Configuration File Format

The pipeline is driven by a JSON configuration file with the following structure:

```json
{
  "target_variable": "species",
  "prediction_type": "classification",
  "features": [
    {
      "name": "sepal_length",
      "imputation_method": "mean"
    },
    ...
  ],
  "feature_reduction": {
    "method": "pca",
    "n_components": 2
  },
  "algorithms": [
    {
      "name": "logistic_regression",
      "is_selected": true,
      "hyperparameters": {
        "C": [0.1, 1.0, 10.0],
        "penalty": ["l2"],
        ...
      }
    },
    ...
  ]
}
```

### Key Configuration Elements

- `target_variable`: The column to predict
- `prediction_type`: Either "classification" or "regression"
- `features`: List of features with their imputation methods
- `feature_reduction`: Method and parameters for feature reduction
- `algorithms`: List of algorithms to try with their hyperparameters

## Extending the Pipeline

The pipeline is designed to be easily extended:

1. Add new imputation methods in `preprocessing.py`
2. Add new feature reduction techniques in `feature_reduction.py`
3. Add new ML algorithms in `model_builder.py`
4. Add new evaluation metrics in `evaluation.py`

## License

This project is open source and available under the MIT License.