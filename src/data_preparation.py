import os
import logging
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, Any

def setup_logging():
    """Set up logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    return logger

def load_config(config_path='src/config.yaml'):
    """Load configuration from YAML file."""
    logger = setup_logging()
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def load_data(data_path='data/data.csv'):
    """Load data from CSV file."""
    logger = setup_logging()
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully with shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def convert_to_numeric(data, numerical_features):
    """Convert numerical features to numeric type."""
    logger = setup_logging()
    for col in numerical_features:
        try:
            # Replace any non-numeric values with NaN
            data[col] = pd.to_numeric(data[col], errors='coerce')
            logger.info(f"Converted {col} to numeric type")
        except Exception as e:
            logger.warning(f"Could not convert {col} to numeric type: {str(e)}")
    return data

def create_preprocessing_pipeline(config):
    """Create preprocessing pipeline for numerical and categorical features."""
    logger = setup_logging()
    try:
        numerical_features = config['data']['numerical_features']
        categorical_features = config['data']['categorical_features']
        
        # Create preprocessing pipelines for numerical and categorical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=config['preprocessing']['numerical']['strategy'])),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(
                strategy=config['preprocessing']['categorical']['strategy'],
                fill_value=config['preprocessing']['categorical']['fill_value']
            )),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        logger.info("Preprocessing pipeline created successfully")
        return preprocessor
    except Exception as e:
        logger.error(f"Error creating preprocessing pipeline: {str(e)}")
        raise

def prepare_data(data, config):
    """Prepare data for model training."""
    logger = setup_logging()
    try:
        # Drop unnecessary columns
        if config['data']['columns_to_drop']:
            data = data.drop(columns=config['data']['columns_to_drop'])
        
        # Convert target to numeric and handle NaN values
        data[config['data']['target_column']] = pd.to_numeric(data[config['data']['target_column']], errors='coerce')
        
        # Remove rows where target is NaN
        data = data.dropna(subset=[config['data']['target_column']])
        logger.info(f"Removed rows with NaN in target variable. New shape: {data.shape}")
        
        # Convert numerical features to numeric type
        data = convert_to_numeric(data, config['data']['numerical_features'])
        
        # Split features and target
        X = data.drop(columns=[config['data']['target_column']])
        y = data[config['data']['target_column']]
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state'],
            stratify=y
        )
        
        logger.info(f"Data split into train ({X_train.shape}) and test ({X_test.shape}) sets")
        
        # Create preprocessing pipeline
        preprocessor = create_preprocessing_pipeline(config)
        
        # Fit and transform the data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        logger.info("Data preprocessing completed successfully")
        
        return X_train_processed, X_test_processed, y_train, y_test, preprocessor
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

if __name__ == "__main__":
    # Test the data preparation pipeline
    config = load_config()
    data = load_data()
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(data, config) 