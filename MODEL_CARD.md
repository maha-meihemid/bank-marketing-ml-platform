# Model Card — Bank Marketing Subscription Classifier

## Model

This model predicts the probability that a customer subscribes to a bank term deposit.

- **Task:** Binary classification
- **Model:** XGBoost
- **Input features:** 24
- **Target:** `y`
- **Primary metric:** ROC-AUC
- **Final test ROC-AUC:** **0.9671**

The model is stored as a complete sklearn pipeline containing the preprocessing transformations and the XGBoost classifier.

---

## Data

The model was trained on the Kaggle Playground Series S5E8 Bank Marketing dataset.

The labelled dataset contains **750,000 observations**, with approximately **12.1% positive examples**.

It was split using stratified sampling:

| Split | Rows |
|---|---:|
| Train | 524,999 |
| Validation | 112,501 |
| Test | 112,500 |

The test set was kept separate during model selection and hyperparameter tuning.

---

## Features

The model uses **16 original features**:

```text
age, job, marital, education, default, balance,
housing, loan, contact, day, month, duration,
campaign, pdays, previous, poutcome
```

I added **8 engineered features**:

```text
has_previous_contact
is_previous_success
is_cellular_contact
balance_per_campaign
duration_per_campaign
log_duration
log_balance_abs
age_group
```

This gives **24 features** in the final model.

Categorical features are one-hot encoded, numerical features are standardized, and binary engineered features are passed directly to the model.

---

## Model Selection

I compared three model families using both the original and engineered feature sets.

| Model | Features | ROC-AUC |
|---|---|---:|
| Logistic Regression | Raw | 0.9429 |
| Logistic Regression | Engineered | 0.9460 |
| Random Forest | Raw | 0.9612 |
| Random Forest | Engineered | 0.9632 |
| XGBoost | Raw | 0.9662 |
| **XGBoost** | **Engineered** | **0.9663** |

The engineered features clearly helped Logistic Regression and Random Forest.

For XGBoost, the improvement was very small:

```text
Raw        : 0.966233
Engineered : 0.966318
```

This suggests that XGBoost was already able to learn most of these nonlinear relationships from the original variables.

XGBoost with engineered features was nevertheless the best baseline and was selected for tuning.

---

## Hyperparameter Tuning

I tuned XGBoost using:

- `RandomizedSearchCV`
- 12 parameter configurations
- 3-fold stratified cross-validation
- ROC-AUC as the optimization metric

The selected parameters were:

```json
{
  "n_estimators": 500,
  "max_depth": 6,
  "learning_rate": 0.08,
  "min_child_weight": 1,
  "subsample": 1.0,
  "colsample_bytree": 0.8,
  "reg_alpha": 0.1,
  "reg_lambda": 1.0
}
```

Best cross-validation ROC-AUC:

```text
0.966921
```

Validation ROC-AUC after tuning:

```text
0.967347
```

The baseline was `0.966318`, so tuning provided a small but measurable improvement.

---

## Final Performance

The tuned model was evaluated once on the held-out test set.

| Metric | Validation | Test |
|---|---:|---:|
| ROC-AUC | 0.967347 | **0.967145** |
| PR-AUC | 0.804548 | **0.800288** |
| Precision | 0.766415 | **0.762013** |
| Recall | 0.672512 | **0.665954** |
| F1 | 0.716399 | **0.710753** |

Precision, recall and F1 use a threshold of `0.50`.

The difference between validation and test ROC-AUC is only:

```text
0.967347 - 0.967145 = 0.000202
```

The final performance is therefore very close to the validation performance.

---

## Important Limitation

The most important limitation is the `duration` feature.

`duration` represents the duration of the current marketing call and is only known once the call has taken place.

Therefore, this model should **not** be interpreted as a pre-contact model answering:

> Which customers should the bank call?

For that use case, `duration` would not be available at prediction time.

The feature is retained here because the project follows the Kaggle competition setting.

A real pre-contact model should be retrained using only features available before the customer is contacted.

---

## Possible Improvements

If I wanted to push the modelling further, I would test:

1. **CatBoost**, to compare native categorical feature handling with XGBoost + one-hot encoding.
2. **A larger XGBoost search**, since the current tuning intentionally tested only 12 configurations.
3. **Feature ablation**, to determine which engineered features actually contribute to performance.
4. **Additional targeted interactions**, rather than creating a large number of arbitrary features.
5. **A simple ensemble**, for example XGBoost + another complementary model, only if it provides a meaningful gain.
6. **A pre-contact version of the model**, excluding features unavailable at decision time.

More aggressive Kaggle techniques such as large stacking ensembles, pseudo-labeling or extensive model blending were intentionally left out because the objective of this project is a maintainable ML pipeline rather than leaderboard optimization.