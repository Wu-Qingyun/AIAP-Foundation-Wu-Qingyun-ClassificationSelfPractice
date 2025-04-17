import os
import logging
import yaml
import numpy as np
from src.data_preparation import load_config, load_data, prepare_data
from src.model_training import train_and_evaluate_models

def setup_logging():
    """Set up logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    return logger

def create_output_dirs(config):
    """Create output directories if they don't exist."""
    logger = setup_logging()
    try:
        for dir_path in config['output'].values():
            os.makedirs(dir_path, exist_ok=True)
        logger.info("Output directories created successfully")
    except Exception as e:
        logger.error(f"Error creating output directories: {str(e)}")
        raise

def main():
    """Main function to run the machine learning pipeline."""
    logger = setup_logging()
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config()
        
        # Create output directories
        logger.info("Creating output directories...")
        create_output_dirs(config)
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        data = load_data(config['data']['input_file'])
        X_train, X_test, y_train, y_test, preprocessor = prepare_data(data, config)
        
        # Save preprocessed data
        logger.info("Saving preprocessed data...")
        os.makedirs('data/processed', exist_ok=True)
        np.save('data/processed/X_train.npy', X_train)
        np.save('data/processed/X_test.npy', X_test)
        np.save('data/processed/y_train.npy', y_train)
        np.save('data/processed/y_test.npy', y_test)
        
        # Train and evaluate models
        logger.info("Training and evaluating models...")
        results = train_and_evaluate_models(X_train, X_test, y_train, y_test, config)
        
        # Save results
        logger.info("Saving results...")
        results.to_csv(os.path.join(config['output']['metrics_dir'], 'model_comparison.csv'), index=False)
        
        logger.info("Pipeline completed successfully!")
        print("\nModel Evaluation Results:")
        print(results)
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 