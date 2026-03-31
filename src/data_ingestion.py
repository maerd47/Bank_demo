import pandas as pd
import os
import numpy as np

def load_data():
    df = pd.read_csv('data/bank-full.csv', sep=';')
    print("Data loaded .......")
    
    return df

def main():
    df = load_data()
    print(f"Data shape: {df.shape}")

if __name__ == '__main__':
    main()

