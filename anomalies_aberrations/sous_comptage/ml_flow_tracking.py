import mlflow
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.set_tracking_uri("URI_MLFLOW")
mlflow.set_experiment("MLflow Quickstart")

with mlflow.start_run(run_name="test_logit_dataset_iris"):
    mlflow.set_tag("mlflow.note.content", """
    Test classification .
    - Données : dataset iris open source de scikit learn
    - Objectif : classer les individus sur la base des features 
    - Params : train/test split avec une proportion 80/20
    - Observation : juste pour l'experience
    """)
    # Load the Iris dataset
    X, y = datasets.load_iris(return_X_y=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model hyperparameters
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 8888,
    }
    # Enable autologging for scikit-learn
    mlflow.sklearn.autolog()
    # Just train the model normally
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)