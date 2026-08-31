# Bank Marketing ML Platform

[![CI](https://github.com/maha-meihemid/bank-marketing-ml-platform/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/maha-meihemid/bank-marketing-ml-platform/actions/workflows/ci.yml)

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
```

## Run the inference API with Docker

Build the production inference image:

```bash
docker build -t bank-marketing-api .
```

Run the container locally:

```bash
docker run --rm -p 8000:8000 bank-marketing-api
```

The health endpoint is available at `http://localhost:8000/health` and the
interactive OpenAPI documentation at `http://localhost:8000/docs`.

## Continuous integration

Pull requests and pushes targeting `develop` or `master` automatically run
Ruff, the complete Python test suite, a Docker image build, and containerized
smoke tests against the health and prediction endpoints.

Run the same Docker smoke test locally from Git Bash:

```bash
bash scripts/ci/docker_smoke_test.sh
```
