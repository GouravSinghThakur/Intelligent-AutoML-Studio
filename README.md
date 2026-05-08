# 🤖 AutoML Studio

An industry-grade AutoML platform that automates the entire machine-learning lifecycle — from raw data upload to hyperparameter-optimised model deployment.

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Auto Task Detection** | Automatically detects Classification vs Regression from target column |
| **14 Algorithms** | 7 classification + 7 regression models with Optuna-tuned hyperparameters |
| **Smart Preprocessing** | Auto-imputation, label encoding, duplicate removal |
| **One-Click Export** | Download the best pipeline as a `.joblib` file |
| **Interactive Visualisations** | Plotly-powered dark-mode charts (ROC, confusion matrix, residuals) |
| **Instant Predictions** | Real-time inference with probability breakdowns |

## 🧠 Supported Algorithms

### Classification
Logistic Regression · SVM · Random Forest · XGBoost · K-Nearest Neighbours · Gradient Boosting · Extra Trees

### Regression
Ridge Regression · SVR · Random Forest Regressor · XGBoost Regressor · KNN Regressor · Gradient Boosting Regressor · Extra Trees Regressor

## 📂 Project Structure

```
AutoML/
├── app.py                          # Slim entry point
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml                 # Theme & server config
├── src/                            # Core business logic
│   ├── __init__.py
│   ├── config.py                   # Centralised settings
│   ├── data_processor.py           # Loading, validation, preprocessing
│   ├── models/
│   │   ├── __init__.py             # Re-exports
│   │   ├── registry.py             # 14 model configs + task detection
│   │   ├── metrics.py              # Classification & regression metrics
│   │   └── training.py             # GridSearchCV, Optuna, auto_train
│   └── visualisations/
│       ├── __init__.py             # Re-exports
│       ├── common.py               # Shared charts (heatmap, bar, pie)
│       ├── classification.py       # ROC curves, confusion matrices
│       └── regression.py           # Actual vs predicted, residuals
└── ui/                             # Streamlit UI layer
    ├── __init__.py
    ├── styles.py                   # CSS injection
    ├── sidebar.py                  # Navigation & status
    ├── helpers.py                  # Shared UI utilities
    └── pages/
        ├── __init__.py
        ├── home.py                 # Landing page
        ├── data_upload.py          # Upload & analyse data
        ├── model_training.py       # Train & export models
        ├── visualisation.py        # Charts & comparisons
        └── prediction.py           # Real-time inference
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## ⚙️ Configuration

All settings are centralised in `src/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `TEST_SIZE` | 0.20 | Train/test split ratio |
| `CV_FOLDS` | 5 | Cross-validation folds |
| `OPTUNA_TRIALS` | 10 | Optuna trials per model |
| `RANDOM_STATE` | 42 | Reproducibility seed |

## 🛠️ Tech Stack

- **Streamlit** — Interactive UI
- **scikit-learn** — ML pipelines & preprocessing
- **XGBoost** — Gradient boosting (GPU-accelerated)
- **Optuna** — Bayesian hyperparameter optimisation
- **Plotly / Matplotlib / Seaborn** — Visualisations
