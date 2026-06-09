import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('car_price_dataset_preprocessing.csv')
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# autolog
mlflow.sklearn.autolog()

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Model Score: {score}")

print("Logging Basic_RandomForest_Autolog selesai!")