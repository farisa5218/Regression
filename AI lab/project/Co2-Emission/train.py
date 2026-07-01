import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

def train_and_evaluate():
    # 1. Load data
    csv_path = 'FuelConsumptionCo2.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find dataset at {csv_path}")
    
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # Define features and target
    target = 'CO2EMISSIONS'
    categorical_features = ['FUELTYPE', 'VEHICLECLASS', 'TRANSMISSION']
    numerical_features = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']
    
    X = df[numerical_features + categorical_features]
    y = df[target]
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Define Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    # 4. Define Models to compare
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    best_r2 = -float('inf')
    best_model_name = None
    best_pipeline = None
    
    print("\nComparing regression models:")
    print(f"{'Model':<20} | {'MAE':<10} | {'RMSE':<10} | {'R2 Score':<10} | {'MAPE (%)':<10}")
    print("-" * 65)
    
    for name, model in models.items():
        # Build pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Fit model
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        results[name] = {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape, 'pipeline': pipeline}
        
        print(f"{name:<20} | {mae:<10.3f} | {rmse:<10.3f} | {r2:<10.5f} | {mape:<10.3f}")
        
        # Keep track of the best model based on R2 Score
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline

    print(f"\nBest Model: {best_model_name} (R2 = {best_r2:.5f})")
    
    # Fit the best model on the entire dataset for final deployment
    print(f"Refitting best model ({best_model_name}) on the complete dataset...")
    best_pipeline.fit(X, y)
    
    # Save the pipeline to model.pkl
    print("Saving model pipeline to model.pkl...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(best_pipeline, f)
    print("Model pipeline saved successfully!")
    
    # 5. Create static image directories if they do not exist
    os.makedirs(os.path.join('static', 'images'), exist_ok=True)
    
    # 6. Generate plots using the test predictions of the best model
    best_model_test_pipeline = results[best_model_name]['pipeline']
    y_pred_test = best_model_test_pipeline.predict(X_test)
    
    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.6, color='#667eea')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f'Actual vs Predicted CO2 Emissions ({best_model_name})', fontsize=14)
    plt.xlabel('Actual CO2 Emissions (g/km)', fontsize=12)
    plt.ylabel('Predicted CO2 Emissions (g/km)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'images', 'actual_vs_predicted.png'), dpi=150)
    plt.close()
    
    # Plot 2: Residuals Plot
    plt.figure(figsize=(8, 6))
    residuals = y_test - y_pred_test
    sns.scatterplot(x=y_pred_test, y=residuals, alpha=0.6, color='#764ba2')
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.title(f'Residuals vs Predicted Values ({best_model_name})', fontsize=14)
    plt.xlabel('Predicted CO2 Emissions (g/km)', fontsize=12)
    plt.ylabel('Residuals (g/km)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'images', 'residuals.png'), dpi=150)
    plt.close()
    
    # Plot 3: Feature Importance (for Random Forest or XGBoost or Decision Tree)
    plt.figure(figsize=(10, 8))
    model_obj = best_pipeline.named_steps['model']
    
    # Get feature names after one-hot encoding
    ohe = best_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    ohe_features = list(ohe.get_feature_names_out(categorical_features))
    feature_names = numerical_features + ohe_features
    
    if hasattr(model_obj, 'feature_importances_'):
        importances = model_obj.feature_importances_
        # Sort and take top 15 features for clarity
        indices = np.argsort(importances)[::-1][:15]
        top_importances = importances[indices]
        top_features = [feature_names[i] for i in indices]
        
        sns.barplot(x=top_importances, y=top_features, hue=top_features, legend=False, palette='viridis')
        plt.title('Top 15 Feature Importances', fontsize=14)
        plt.xlabel('Relative Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join('static', 'images', 'feature_importance.png'), dpi=150)
    elif hasattr(model_obj, 'coef_'):
        # For linear models, use absolute coefficients
        coefs = np.abs(model_obj.coef_)
        indices = np.argsort(coefs)[::-1][:15]
        top_coefs = coefs[indices]
        top_features = [feature_names[i] for i in indices]
        
        sns.barplot(x=top_coefs, y=top_features, hue=top_features, legend=False, palette='mako')
        plt.title('Top 15 Feature Coefficients (Absolute)', fontsize=14)
        plt.xlabel('Absolute Coefficient Value', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join('static', 'images', 'feature_importance.png'), dpi=150)
    
    plt.close()
    print("Visualizations saved under static/images/")
    
    # Write a quick text summary of results for other steps to read if needed
    with open('model_metrics.txt', 'w') as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"R2 Score: {best_r2:.5f}\n")
        for name, metrics in results.items():
            f.write(f"{name} -> MAE: {metrics['mae']:.3f}, RMSE: {metrics['rmse']:.3f}, R2: {metrics['r2']:.5f}, MAPE: {metrics['mape']:.3f}\n")

if __name__ == '__main__':
    train_and_evaluate()
