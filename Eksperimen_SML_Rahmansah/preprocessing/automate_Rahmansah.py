import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(input_path, output_path):
    # load dataset
    df = pd.read_csv(input_path)

    # Validasi Price
    if df['Price'].isnull().any():
        raise ValueError("Data Price mengandung nilai kosong!")
    if (df['Price'] < 0).any():
        raise ValueError("Data Price mengandung nilai negatif!")

    # Pisahkan Fitur dan Target 
    features = df.drop(columns=['Price'])
    target = df['Price']

    # Imputasi hanya pada fitur 
    num_cols = features.select_dtypes(include=[np.number]).columns
    cat_cols = features.select_dtypes(include=['object']).columns
    
    features[num_cols] = SimpleImputer(strategy='median').fit_transform(features[num_cols])
    features[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(features[cat_cols])

    # Gabungkan kembali untuk proses berikutnya
    df = pd.concat([features, target], axis=1)

    # hapus duplikat
    df = df.drop_duplicates()

    # Outlier handling (Price)
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['Price'] >= Q1 - 1.5 * IQR) & (df['Price'] <= Q3 + 1.5 * IQR)]

    # Feature Engineering
    df = df.drop(columns=['Car_ID'])
    df['Car_Age'] = 2026 - df['Model_Year']
    df['Mileage_per_Year'] = df['Mileage'] / (df['Car_Age'] + 1)

    # Binning
    bins = [2000, 2010, 2015, 2020, 2026]
    labels = ['Vintage', 'Old', 'Modern', 'New']
    df['Year_Category'] = pd.cut(df['Model_Year'], bins=bins, labels=labels, include_lowest=True)

    # Encoding Data Kategorikal
    df = pd.get_dummies(df, columns=['Fuel_Type', 'Transmission', 'Year_Category', 'Brand'], drop_first=True)

    # Normalisasi / Standarisasi Fitur
    scaler = StandardScaler()
    num_cols_to_scale = ['Mileage', 'Horsepower', 'Engine_Size', 'Car_Age', 'Mileage_per_Year']
    df[num_cols_to_scale] = scaler.fit_transform(df[num_cols_to_scale])

    # Simpan dataset siap pakai
    df.to_csv(output_path, index=False)
    print(f"Preprocessing selesai! Data disimpan di: {output_path}")

if __name__ == "__main__":
    input_file = '../car_price_dataset.csv'
    output_file = 'car_price_dataset_preprocessing_automated.csv'
    preprocess_data(input_file, output_file)

# Testing workflow trigger