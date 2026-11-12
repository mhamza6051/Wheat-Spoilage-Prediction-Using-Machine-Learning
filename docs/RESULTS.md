# Detailed Results & Experiment Notes

All numbers are taken directly from the experiment notebook
(`notebooks/wheat_spoilage_analysis.ipynb`). Split: 70% train / 30% test
(1,200 test samples), `random_state=42`, features scaled with `StandardScaler`
fitted on the training set only.

## Data generation

- 4,000 samples, balanced 2,000 Safe / 2,000 Spoiled, generated with `np.random.seed(42)`.
- Per-class Gaussian profiles — Safe: humidity N(47, 8), gases N(80, 40), temperature N(21, 5);
  Spoiled: humidity N(65, 12), gases N(200, 80), temperature N(32, 7).
- 20% of samples were overwritten with ambiguous mid-range values
  (humidity ~55, gases ~150, temperature ~28) so the classes genuinely overlap.
- Consequence visible in the raw data: ranges exceed physical sensor limits in the tails
  (e.g. humidity up to 102%, negative gas readings) — normal for unclipped Gaussian simulation,
  and one reason real-data validation is listed as future work in the README.

## Model configurations

| Model | Key settings |
|---|---|
| Logistic Regression | C=0.3, penalty L2, solver saga, max_iter 1000 |
| Decision Tree | max_depth 3, min_samples_split 20, min_samples_leaf 15 |
| Random Forest | n_estimators 50, max_depth 4, min_samples_split 15, min_samples_leaf 10, max_features 2 |

All three were deliberately capacity-limited (shallow trees, strong LR regularization)
to keep the comparison honest on overlapping data.

## Classification reports (test set)

**Logistic Regression — accuracy 0.85**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Safe (0) | 0.87 | 0.85 | 0.86 | 616 |
| Spoiled (1) | 0.84 | 0.86 | 0.85 | 584 |

**Decision Tree — accuracy 0.84**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Safe (0) | 0.84 | 0.84 | 0.84 | 616 |
| Spoiled (1) | 0.83 | 0.83 | 0.83 | 584 |

**Random Forest — accuracy 0.86 (best)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Safe (0) | 0.86 | 0.86 | 0.86 | 616 |
| Spoiled (1) | 0.85 | 0.85 | 0.85 | 584 |

The notebook also includes an accuracy comparison chart, confusion matrices for all
three models, and ROC curves with AUC for each classifier.

## Conclusion

Random Forest wins by a small, consistent margin (0.86 vs 0.85 / 0.84), with balanced
errors across both classes — no model is simply predicting the majority pattern.
Because the classes overlap by design, these scores represent close to the practical
ceiling for three features on this data; the honest improvement path is better
features and real sensor data, not bigger models.
