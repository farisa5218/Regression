# =============================================================================
# train.py — Model Training Script for CO2 Emission Prediction
# =============================================================================
# What this script does:
#   1. Loads the dataset (FuelConsumptionCo2.csv).
#   2. Splits data into training (80%) and testing (20%) sets.
#   3. Builds a preprocessing pipeline (scaling + encoding).
#   4. Trains 5 regression models and compares their performance.
#   5. Saves the best model to model.pkl for use by the web app.
#   6. Generates 3 diagnostic charts under static/images/.
#
# Run this script once (or whenever you want to retrain the model):
#   >> python train.py
# =============================================================================

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


# ── Constants ─────────────────────────────────────────────────────────────────
DATASET_PATH   = 'FuelConsumptionCo2.csv'
MODEL_PATH     = 'model.pkl'
METRICS_PATH   = 'model_metrics.txt'
IMAGES_DIR     = os.path.join('static', 'images')

TARGET              = 'CO2EMISSIONS'
NUMERICAL_FEATURES  = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']
CATEGORICAL_FEATURES = ['FUELTYPE', 'VEHICLECLASS', 'TRANSMISSION']

# All 5 models to compare
MODELS = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression':  Ridge(alpha=1.0),
    'Decision Tree':     DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest':     RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
    'XGBoost':           XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
}


# ── Helper Functions ──────────────────────────────────────────────────────────
def mean_absolute_percentage_error(y_true, y_pred):
    """Custom MAPE metric — ignores zero-value rows to avoid division by zero."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def build_preprocessor():
    """
    Returns a ColumnTransformer that:
      - Standardizes numerical columns (mean=0, std=1).
      - One-hot encodes categorical columns.
    """
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(),                                         NUMERICAL_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES),
    ])


# ── Plotting Functions ────────────────────────────────────────────────────────
def plot_actual_vs_predicted(y_test, y_pred, model_name):
    """Scatter plot: Actual vs Predicted CO2 values. Points should lie on the red line."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='#667eea')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect prediction')
    plt.title(f'Actual vs Predicted CO2 Emissions\n({model_name})', fontsize=14)
    plt.xlabel('Actual CO2 (g/km)')
    plt.ylabel('Predicted CO2 (g/km)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'actual_vs_predicted.png'), dpi=150)
    plt.close()


def plot_residuals(y_test, y_pred, model_name):
    """Residual plot: Should be randomly scattered around 0 for a good model."""
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, color='#764ba2')
    plt.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero error line')
    plt.title(f'Residuals vs Predicted Values\n({model_name})', fontsize=14)
    plt.xlabel('Predicted CO2 (g/km)')
    plt.ylabel('Residual / Error (g/km)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'residuals.png'), dpi=150)
    plt.close()


def plot_feature_importance(pipeline, model_name):
    """Bar chart of the most important features. Only works for tree-based or linear models."""
    model_obj = pipeline.named_steps['model']
    ohe       = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    all_features = NUMERICAL_FEATURES + list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))

    plt.figure(figsize=(10, 7))

    if hasattr(model_obj, 'feature_importances_'):
        # Tree-based models (Random Forest, XGBoost, Decision Tree)
        importances = model_obj.feature_importances_
        title = 'Top 15 Feature Importances'
        xlabel = 'Relative Importance'
    elif hasattr(model_obj, 'coef_'):
        # Linear models — use absolute coefficient values
        importances = np.abs(model_obj.coef_)
        title = 'Top 15 Feature Coefficients (Absolute)'
        xlabel = 'Absolute Coefficient'
    else:
        print(f"  [{model_name}] Feature importance not available for this model type.")
        plt.close()
        return

    top_idx   = np.argsort(importances)[::-1][:15]
    top_vals  = importances[top_idx]
    top_names = [all_features[i] for i in top_idx]

    sns.barplot(x=top_vals, y=top_names, hue=top_names, legend=False, palette='viridis')
    plt.title(f'{title}\n({model_name})', fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'feature_importance.png'), dpi=150)
    plt.close()


# ── Main Training Function ────────────────────────────────────────────────────
def train_and_evaluate():
    # ── 1. Load Dataset ───────────────────────────────────────────────────────
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: '{DATASET_PATH}'\n"
                                f"Run prepare_real_dataset.py first to generate it.")
    print(f"Loading dataset from '{DATASET_PATH}'...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  {len(df):,} rows loaded.\n")

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # ── 2. Train / Test Split (80% train, 20% test) ───────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── 3. Build Preprocessor ─────────────────────────────────────────────────
    preprocessor = build_preprocessor()

    # ── 4. Train All Models & Compare ────────────────────────────────────────
    print(f"{'Model':<20} | {'MAE':>8} | {'RMSE':>8} | {'R²':>8} | {'MAPE %':>8}")
    print("-" * 65)

    results = {}
    best_r2, best_name, best_pipeline = -float('inf'), None, None

    for name, model in MODELS.items():
        # Each pipeline = preprocessing → model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model),
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        results[name] = dict(mae=mae, rmse=rmse, r2=r2, mape=mape, pipeline=pipeline, y_pred=y_pred)
        print(f"{name:<20} | {mae:>8.3f} | {rmse:>8.3f} | {r2:>8.5f} | {mape:>8.3f}")

        if r2 > best_r2:
            best_r2, best_name, best_pipeline = r2, name, pipeline

    print(f"\n✅ Best Model: {best_name}  (R² = {best_r2:.5f})")

    # ── 5. Refit Best Model on Full Dataset & Save ────────────────────────────
    print(f"\nRefitting '{best_name}' on the full dataset for deployment...")
    best_pipeline.fit(X, y)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_pipeline, f)
    print(f"Model saved to '{MODEL_PATH}'.")

    # ── 6. Save Metrics Summary ───────────────────────────────────────────────
    with open(METRICS_PATH, 'w') as f:
        f.write(f"Best Model: {best_name}\n")
        f.write(f"R² Score:   {best_r2:.5f}\n\n")
        f.write("All Model Results:\n")
        for name, m in results.items():
            f.write(f"  {name:<20} → MAE: {m['mae']:.3f}, RMSE: {m['rmse']:.3f}, "
                    f"R²: {m['r2']:.5f}, MAPE: {m['mape']:.3f}%\n")
    print(f"Metrics saved to '{METRICS_PATH}'.")

    # ── 7. Generate & Save Diagnostic Charts ──────────────────────────────────
    os.makedirs(IMAGES_DIR, exist_ok=True)

    best_test_pipeline = results[best_name]['pipeline']
    y_pred_test        = results[best_name]['y_pred']

    print("\nGenerating charts...")
    plot_actual_vs_predicted(y_test, y_pred_test, best_name)
    print("  ✔ actual_vs_predicted.png")
    plot_residuals(y_test, y_pred_test, best_name)
    print("  ✔ residuals.png")
    plot_feature_importance(best_test_pipeline, best_name)
    print("  ✔ feature_importance.png")

    print(f"\n🎉 Training complete! Charts saved to '{IMAGES_DIR}/'")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    train_and_evaluate()
