import pandas as pd
import os
import numpy as np
from xgboost import XGBClassifier
from preprocess import preprocess

import mlflow 
import mlflow.xgboost

# Connect to remote MLflow server
#mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("my-first-experiment")




def train_model():
    train_df, val_df = preprocess()

    with mlflow.start_run():
    

        X_data = pd.read_csv('bank_data/train.csv')
        train = X_data.drop(columns=['y'])
        y_label = X_data['y']

        # Define the model with hyperparameters
        # 'objective' specifies the learning task and the corresponding learning objective
        # For multi-class classification, use 'multi:softprob' (default for XGBClassifier with multi-class data) or 'multi:softmax'
        
       

        params = {
            "objective": "binary:logistic",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "eval_metric": "logloss"
        }

        model = XGBClassifier(**params)

        # Log parameters
        mlflow.log_params(params)

         # Fit the model to the training data
        model.fit(train, y_label)

        print("Model trained ..................")

        return model

def main():
    model = train_model()
    print("Model training completed.")

if __name__ == '__main__':
    main()