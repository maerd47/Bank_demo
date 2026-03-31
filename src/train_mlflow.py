import pandas as pd
import os
import numpy as np
import mlflow 
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier



# Connect to remote MLflow server
#mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("my-first-experiment")


def run_pipeline():

    # -----------------------------
    # Start MLflow Run
    # -----------------------------
    with mlflow.start_run():

        # ---- Load data ----
        df = pd.read_csv('data/bank-full.csv', sep=';')

        print("Shape:", df.shape)
        print("Target distribution:\n", df['y'].value_counts())

        # ---- Target conversion ----
        df['y'] = df['y'].map({'yes': 1, 'no': 0})

        # ---- Feature separation ----
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numerical_cols.remove('y')

        # ---- Encoding ----
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        df_encoded = df_encoded.replace({True: 1, False: 0}).infer_objects(copy=False)

        # ---- Scaling ----
        scaler = StandardScaler()
        df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

        # ---- Split ----
        X = df_encoded.drop('y', axis=1)
        y = df_encoded['y']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # -----------------------------
        # Model + Hyperparameters
        # -----------------------------
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

        # ---- Train ----
        model.fit(X_train, y_train)

        # ---- Predict ----
        y_pred = model.predict(X_test)

        # ---- Metrics ----
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

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
        mlflow.xgboost.log_model(model, "XGB")

        print("\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")

        


if __name__ == "__main__":
    run_pipeline()