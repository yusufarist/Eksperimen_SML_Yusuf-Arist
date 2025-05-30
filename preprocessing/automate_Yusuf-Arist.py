import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def preprocess_data(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    # Memisahkan fitur dan target
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    # Encoding variabel kategorikal
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Normalisasi fitur numerik
    scaler = StandardScaler()
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Buat direktori output jika belum ada
    os.makedirs('heart_preprocessing', exist_ok=True)
    
    # Path file yang akan diproses (relatif terhadap folder preprocessing)
    input_file = '../heart_raw/heart.csv'
    
    # Proses data
    X_train, X_test, y_train, y_test = preprocess_data(input_file)
    
    # Simpan hasil preprocessing
    X_train.to_csv('heart_preprocessing/X_train.csv', index=False)
    X_test.to_csv('heart_preprocessing/X_test.csv', index=False)
    y_train.to_csv('heart_preprocessing/y_train.csv', index=False)
    y_test.to_csv('heart_preprocessing/y_test.csv', index=False)
