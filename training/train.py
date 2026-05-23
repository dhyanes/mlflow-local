import os
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ==========================================
# MLflow Configuration
# ==========================================

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://192.168.56.10:30900"
os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "password123"
os.environ["AWS_S3_FORCE_PATH_STYLE"] = "true"

mlflow.set_tracking_uri("http://mlflow.local:32318")

mlflow.set_experiment("iris-classification")


# ==========================================
# Load Dataset
# ==========================================

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# Model Parameters
# ==========================================

n_estimators = 100
max_depth = 5

# ==========================================
# Start MLflow Run
# ==========================================

with mlflow.start_run():

    # ======================================
    # Train Model
    # ======================================

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    # ======================================
    # Predictions
    # ======================================

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    # ======================================
    # Log Parameters
    # ======================================

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # ======================================
    # Log Metrics
    # ======================================

    mlflow.log_metric("accuracy", accuracy)

    # ======================================
    # Log Model
    # ======================================

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="iris-randomforest"
    )

    print(f"Model Accuracy: {accuracy}")

    print("Model logged successfully to MLflow")