# Wheat Spoilage Prediction Using Machine Learning

**Predicting wheat grain spoilage from storage conditions — humidity, gas concentration and temperature — with three compared ML classifiers and a live Streamlit app.**

Wheat stored in silos and warehouses spoils when temperature, humidity and microbial gas build-up cross safe limits — often before any visible sign appears. This project builds a machine learning system that reads those three storage-environment signals and predicts whether a wheat batch is **Safe** or **Spoiled**, so storage operators can intervene early.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-ff4b4b)
![Task](https://img.shields.io/badge/Task-Binary%20Classification-green)

---

## What this project does

1. **Builds a realistic dataset** of 4,000 wheat-storage readings (humidity, gases, temperature → spoiled or safe), with deliberately overlapping class distributions and 20% ambiguous borderline cases — because real spoilage does not separate cleanly.
2. **Trains and honestly compares three classifiers** — Logistic Regression, Decision Tree, and Random Forest — on a 70/30 split with StandardScaler feature scaling.
3. **Evaluates properly**: accuracy, per-class precision/recall/F1, confusion matrices, and ROC curves.
4. **Deploys all three trained models** in a Streamlit app where the user sets the three storage readings with sliders, picks a model, and gets an instant spoilage prediction with confidence.

## Results

| Model | Accuracy | F1 (Safe) | F1 (Spoiled) |
|---|---|---|---|
| Logistic Regression (C=0.3, L2, saga) | 0.85 | 0.86 | 0.85 |
| Decision Tree (max depth 3) | 0.84 | 0.84 | 0.83 |
| **Random Forest (50 trees, max depth 4)** | **0.86** | **0.86** | **0.85** |

Test set: 1,200 samples. Accuracies in the mid-80s are **by design, not a shortcoming** — the dataset was built with overlapping distributions and ambiguous cases, so a score near 100% would mean the problem was unrealistically easy. The regularized, depth-limited models were also chosen deliberately to avoid overfitting the simulation.

## Dataset

`data/wheat_spoilage_final_dataset.csv` — 4,000 rows, perfectly balanced (2,000 Safe / 2,000 Spoiled).

| Feature | Unit | Safe profile (mean) | Spoiled profile (mean) |
|---|---|---|---|
| humidity | % | ~47 | ~65 |
| gases | ppm | ~80 | ~200 |
| temperature | °C | ~21 | ~32 |

> **Transparency note:** the dataset is *simulated* — it is generated inside the notebook (`notebooks/wheat_spoilage_analysis.ipynb`, cell 1) from Gaussian distributions modelled on known safe/spoilage storage ranges, with 20% ambiguous cases mixed in. It is a modelling and prototyping dataset for the prediction system, not field sensor recordings. The generation code is included, so the entire pipeline is reproducible end to end.

## Method

```
Simulated sensor data (4,000 × 3 features)
        │
        ├─► EDA — class-wise distribution of humidity / gases / temperature
        ├─► 70/30 train-test split ─► StandardScaler
        ├─► Logistic Regression ─┐
        ├─► Decision Tree        ├─► Accuracy · F1 · Confusion matrix · ROC
        └─► Random Forest       ─┘
        │
        └─► Best models + scaler saved (joblib) ─► Streamlit app
```

**Safe storage ranges used by the app:** humidity 40–55%, gases below ~150 ppm, temperature 15–27 °C. Spoilage risk rises sharply above 60% humidity, 180 ppm gases, or 30 °C.

## The app

`app.py` — **Wheat Spoilage Detection System** (Streamlit):

- Sliders for humidity, gas concentration, and temperature
- A chart comparing your current readings against the safe ranges
- Model selector — switch between Logistic Regression, Decision Tree, and Random Forest
- One-click prediction with safe/spoiled probability and a confidence bar

### Run it locally

```bash
git clone https://github.com/mhamza6051/Wheat-Spoilage-Prediction-Using-Machine-Learning.git
cd Wheat-Spoilage-Prediction-Using-Machine-Learning
pip install -r requirements.txt
streamlit run app.py
```

## Repository structure

```
├── app.py                                  # Streamlit app (loads models from models/)
├── notebooks/
│   └── wheat_spoilage_analysis.ipynb       # Data generation, EDA, training, evaluation
├── models/
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   └── scaler.pkl                          # StandardScaler fitted on the training set
├── data/
│   └── wheat_spoilage_final_dataset.csv    # 4,000-row dataset (simulated, see note above)
├── docs/
│   └── RESULTS.md                          # Full classification reports and experiment notes
├── requirements.txt
└── README.md
```

## Tech stack

Python · scikit-learn · pandas · NumPy · matplotlib · seaborn · joblib · Streamlit

## Limitations & future work

- **Simulated data** — the next step is validating on real silo sensor logs; the pipeline (features → scaler → model) transfers directly.
- **Three features only** — adding moisture content of the grain itself, storage duration, and CO₂-specific sensing should sharpen accuracy.
- Planned: gradient boosting comparison, threshold tuning for higher spoiled-class recall (missing a spoiled batch costs more than a false alarm), and live sensor input instead of sliders.

## Author

**Muhammad Hamza** — Data Scientist / AI-ML Developer
GitHub: [@mhamza6051](https://github.com/mhamza6051)
