import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from model_training import train_model

import mlflow 
import mlflow.xgboost

# Connect to remote MLflow server
#mlflow.set_tracking_uri("http://localhost:5000")
#mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Bank-demo experiment")

def evaluate():
    model = train_model()

    with mlflow.start_run():

        data_test = pd.read_csv('bank_data/validation.csv')
        test = data_test.drop(columns = ['y'])
        test_label = data_test['y']


        # Make predictions on the test data
        y_pred = model.predict(test)

        # Evaluate the model
        accuracy = accuracy_score(test_label, y_pred)
        print(f"Accuracy: {accuracy*100:.2f}%")
        precision = precision_score(test_label, y_pred)
        recall = recall_score(test_label, y_pred)
        f1 = f1_score(test_label, y_pred)
        roc_auc = roc_auc_score(test_label, y_pred)

         # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # ---- Save model locally (for DVC or manual use) ----
        os.makedirs("model", exist_ok=True)
        model.save_model("model/xgboost_model.json")

        # ---- Log model to MLflow ----
        mlflow.xgboost.log_model(model, "model")

        # Evaluation metrics
        print("Accuracy:", accuracy_score(test_label, y_pred))
        print("Precision:", precision_score(test_label, y_pred))
        print("Recall:", recall_score(test_label, y_pred))
        print("F1 Score:", f1_score(test_label, y_pred))
        print("ROC AUC:", roc_auc_score(test_label, y_pred))

        return accuracy

evaluate()
