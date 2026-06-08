import dagshub
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    mean_absolute_error, 
    mean_absolute_percentage_error, 
    root_mean_squared_error # new scikit-learn
)
import os
import matplotlib.pyplot as plt

dagshub.init(repo_owner='rahmansah', repo_name='SMSML_rahmansah', mlflow=True)

# Load data
df = pd.read_csv('../Eksperimen_SML_Rahmansah/preprocessing/car_price_dataset_preprocessing.csv')
X = df.drop('Price', axis=1)
y = df['Price']

# Grid Search setup
params = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
rf = RandomForestRegressor(random_state=42)
grid = GridSearchCV(rf, params, cv=3)
grid.fit(X, y)

# Manual Logging 
mlflow.set_experiment("Skilled_Tuning_Advanced")
with mlflow.start_run(run_name="Skilled_Tuning_Full_Metrics"):
    # Log param terbaik
    mlflow.log_params(grid.best_params_)
    
    # Hitung metrik manual
    y_pred = grid.best_estimator_.predict(X)
    metrics = {
        "MSE": mean_squared_error(y, y_pred),
        "RMSE": root_mean_squared_error(y, y_pred), # <--- Cara baru yang benar
        "MAE": mean_absolute_error(y, y_pred),
        "MAPE": mean_absolute_percentage_error(y, y_pred),
        "R2": r2_score(y, y_pred)
    }
    
    # Log semua metrik 
    mlflow.log_metrics(metrics)
    
    # Log model dengan input  
    mlflow.sklearn.log_model(
        grid.best_estimator_, 
        "model",
        input_example=X.iloc[:1]
    )
    
    # Log artefak dataset
    mlflow.log_artifact("../Eksperimen_SML_Rahmansah/preprocessing/car_price_dataset_preprocessing.csv")

    # artefak: Visualisasi Feature Importance
    feat_importances = pd.Series(grid.best_estimator_.feature_importances_, index=X.columns)
    feat_importances.nlargest(5).plot(kind='barh')
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")

print("Skilled_Tuning_Advanced manual logging selesai!")