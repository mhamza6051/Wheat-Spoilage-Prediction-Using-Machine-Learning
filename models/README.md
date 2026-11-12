# Trained Models

Trained with scikit-learn and saved with joblib by `notebooks/wheat_spoilage_analysis.ipynb`.
All models take StandardScaler-scaled inputs in the order **[humidity, gases, temperature]**
and output class `0` = Safe, `1` = Spoiled, plus class probabilities.

| File | Model | Test accuracy |
|---|---|---|
| `logistic_regression_model.pkl` | Logistic Regression (C=0.3, L2, saga) | 0.85 |
| `decision_tree_model.pkl` | Decision Tree (max_depth 3) | 0.84 |
| `random_forest_model.pkl` | Random Forest (50 trees, max_depth 4) | 0.86 |
| `scaler.pkl` | StandardScaler fitted on the training split | — |

`app.py` loads all four files automatically from this folder.
