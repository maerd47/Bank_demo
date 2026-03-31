import mlflow

# Connect to remote MLflow server
#mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("my-first-experiment")


# Register the Model
model_name = 'XGB'
run_id= "6725bfc626ba41c3af2d2437481914b3"
model_uri = f'runs:/{run_id}/{model_name}'

with mlflow.start_run(run_id=run_id):
    mlflow.register_model(model_uri=model_uri, name=model_name)

# Transition the Model to Production
model_version = 1
current_model_uri = f"models:/{model_name}/{model_version}"
production_model_name = "bank-data-prod"

client = mlflow.MlflowClient()
client.copy_model_version(src_model_uri=current_model_uri, dst_name=production_model_name)
        