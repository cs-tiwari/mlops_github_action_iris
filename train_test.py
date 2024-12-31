import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import mlflow
import mlflow.sklearn
import os
import joblib

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Set up MLflow experiment
mlflow.set_experiment("Iris Dataset Experiment1")

with mlflow.start_run():
    # Load the Iris dataset
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Split into train and test sets
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Log parameters, metrics, and model to MLflow
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    # Save the model locally (optional)
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/random_forest_model.pkl")
