import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv('../Eksperimen_SML_Rahmansah/preprocessing/car_price_dataset_preprocessing.csv')
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Konfigurasi MLflow
mlflow.set_experiment("Basic_RandomForest_Autolog")

mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True
)

with mlflow.start_run(run_name="Basic_RandomForest_Autolog"):
    # Kita tambahkan tag agar mudah dicari di dashboard nanti
    mlflow.set_tag("developer", "Rahmansah")
    mlflow.set_tag("model_type", "RandomForest_Basic")
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"Model Score: {score}")

print("Logging Basic_RandomForest_Autolog selesai!")