import os
import logging
import yaml
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def setup_logging():
    """Set up logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    return logger

def get_model_configs(config):
    """Get model configurations from config file."""
    logger = setup_logging()
    try:
        model_configs = {
            'logistic_regression': config['models']['logistic_regression'],
            'random_forest': config['models']['random_forest']
        }
        logger.info("Model configurations loaded successfully")
        return model_configs
    except Exception as e:
        logger.error(f"Error loading model configurations: {str(e)}")
        raise

def tune_hyperparameters(model, param_grid, X_train, y_train):
    """Tune hyperparameters using GridSearchCV."""
    logger = setup_logging()
    try:
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    except Exception as e:
        logger.error(f"Error tuning hyperparameters: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test, metrics):
    """Evaluate model performance using specified metrics."""
    logger = setup_logging()
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results = {}
        for metric in metrics:
            if metric == 'accuracy':
                results[metric] = accuracy_score(y_test, y_pred)
            elif metric == 'precision':
                results[metric] = precision_score(y_test, y_pred)
            elif metric == 'recall':
                results[metric] = recall_score(y_test, y_pred)
            elif metric == 'f1':
                results[metric] = f1_score(y_test, y_pred)
            elif metric == 'roc_auc':
                results[metric] = roc_auc_score(y_test, y_pred_proba)
        
        logger.info("Model evaluation completed successfully")
        return results
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

def save_model(model, model_name, metrics, config):
    """Save model and metrics to disk."""
    logger = setup_logging()
    try:
        # Create directories if they don't exist
        os.makedirs(config['output']['models_dir'], exist_ok=True)
        os.makedirs(config['output']['metrics_dir'], exist_ok=True)
        
        # Save model
        model_path = os.path.join(config['output']['models_dir'], f'{model_name}.joblib')
        joblib.dump(model, model_path)
        
        # Save metrics
        metrics_path = os.path.join(config['output']['metrics_dir'], f'{model_name}_metrics.csv')
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        
        logger.info(f"Model and metrics saved successfully for {model_name}")
    except Exception as e:
        logger.error(f"Error saving model and metrics: {str(e)}")
        raise

def train_and_evaluate_models(X_train, X_test, y_train, y_test, config):
    """Train and evaluate multiple models."""
    logger = setup_logging()
    try:
        model_configs = get_model_configs(config)
        results = []
        
        for model_name, params in model_configs.items():
            logger.info(f"Training {model_name}...")
            
            # Initialize model
            if model_name == 'logistic_regression':
                model = LogisticRegression(**params)
            elif model_name == 'random_forest':
                model = RandomForestClassifier(**params)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            metrics = evaluate_model(model, X_test, y_test, config['evaluation']['metrics'])
            
            # Save model and metrics
            save_model(model, model_name, metrics, config)
            
            # Add results to list
            results.append({
                'model': model_name,
                **metrics
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        logger.info("All models trained and evaluated successfully")
        return results_df
    except Exception as e:
        logger.error(f"Error in model training and evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    # Test the model training pipeline
    with open('src/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load preprocessed data
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, config)
    print("\nModel Evaluation Results:")
    print(results) 