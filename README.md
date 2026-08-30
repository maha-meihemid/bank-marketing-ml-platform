# Bank Marketing ML Platform

End-to-end machine learning engineering project for binary classification on the Kaggle Bank Marketing dataset.

## Objective

Build a reproducible ML pipeline covering:

- data ingestion
- data validation
- preprocessing
- feature engineering
- model training
- experiment tracking
- model evaluation
- API serving
- containerization
- CI/CD
- AWS deployment

## Models

The project compares:

- Logistic Regression
- Random Forest
- XGBoost

## Architecture

```text
Kaggle
  |
  v
Data ingestion
  |
  v
Pandera validation
  |
  v
Preprocessing
  |
  v
Feature engineering
  |
  v
Model training
  |
  v
MLflow
  |
  v
FastAPI
  |
  v
Docker
  |
  v
AWS