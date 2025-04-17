# Coronary Artery Disease Prediction

This project implements a machine learning pipeline to predict the survival of coronary artery disease patients using various clinical features.

## Project Structure

```
root/
    |- eda.ipynb              # Exploratory Data Analysis notebook
    |- README.md              # This file
    |- requirements.txt       # Project dependencies
    |- main.py               # Main pipeline execution script
    |- data/
        |- data.csv          # Input dataset
    |- src/
        |- data_preparation.py  # Data preprocessing module
        |- model_training.py    # Model training and evaluation module
        |- config.yaml          # Configuration file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the exploratory data analysis:
   - Open `eda.ipynb` in Jupyter Notebook/Lab
   - Execute all cells to generate visualizations and insights

2. Run the machine learning pipeline:
```bash
python main.py
```

The pipeline will:
- Load and preprocess the data
- Train Logistic Regression and Random Forest models
- Evaluate model performance
- Save models and metrics to respective directories

## Configuration

Model parameters and preprocessing steps can be configured in `src/config.yaml`. 