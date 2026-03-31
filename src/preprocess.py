from py_compile import main

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_ingestion import load_data

def preprocess():
    df = load_data()

    # ---- Step 2: Convert target variable 'y' to binary ----
    df['y'] = df['y'].map({'yes': 1, 'no': 0})

    # ---- Step 3: Identify categorical and numerical columns ----
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols.remove('y')

    # ---- Step 4: One-hot encode categorical variables ----
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df_encoded = df_encoded.replace({True: 1, False: 0})

    # ---- Step 5: Normalize numerical features ----
    scaler = StandardScaler()
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

    # ---- Step 6: Train-test split ----
    X = df_encoded.drop('y', axis=1)
    y = df_encoded['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    # Create a folder to store CSVs
    os.makedirs('bank_data', exist_ok=True)

    # Combine X and y
    train_df = pd.concat([y_train.reset_index(drop=True), X_train.reset_index(drop=True)], axis=1)
    val_df   = pd.concat([y_test.reset_index(drop=True),  X_test.reset_index(drop=True)],  axis=1)

    # Save as CSVs
    train_df.to_csv('bank_data/train.csv', index=False, header=True)
    val_df.to_csv('bank_data/validation.csv', index=False, header=True)
    print("Data processed ...................")

    return train_df, val_df

def main():
    train_df, val_df = preprocess()
    print(f"Training data shape: {train_df.shape}")
    print(f"Validation data shape: {val_df.shape}")

if __name__ == '__main__':
    main()