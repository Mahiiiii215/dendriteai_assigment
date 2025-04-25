"""
Main script for running the ML pipeline.
"""
import os
import sys
import logging
import argparse
import json
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.pipeline import Pipeline

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ml_pipeline.log')
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run ML pipeline based on JSON configuration')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='algoparams_from_ui.json',
        help='Path to JSON configuration file'
    )
    
    parser.add_argument(
        '--data', 
        type=str, 
        default='iris.csv',
        help='Path to CSV data file'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='results.json',
        help='Path to save results'
    )
    
    return parser.parse_args()

def main():
    """Main function to run the ML pipeline."""
    # Set up logging
    logger = setup_logging()
    logger.info("Starting ML Pipeline")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Make sure input files exist
    config_path = Path(args.config)
    data_path = Path(args.data)
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)
    
    try:
        # Initialize and run pipeline
        logger.info(f"Initializing pipeline with config: {config_path} and data: {data_path}")
        pipeline = Pipeline(str(config_path), str(data_path))
        
        logger.info("Running pipeline")
        results = pipeline.run()
        
        # Save results
        output_path = Path(args.output)
        logger.info(f"Saving results to {output_path}")
        pipeline.save_results(str(output_path))
        
        # Print summary
        print("\n=== ML Pipeline Execution Summary ===")
        for algo_name, result in results.items():
            print(f"\nResults for {algo_name}:")
            print(result['formatted_results'])
        
        logger.info("Pipeline execution completed successfully")
        
    except Exception as e:
        logger.error(f"Error in pipeline execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()