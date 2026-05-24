import os
import mlflow
import mlflow.pyfunc
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ==================================================
# MinIO Configuration
# ==================================================

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://192.168.56.10:30900"
os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "password123"
os.environ["AWS_S3_FORCE_PATH_STYLE"] = "true"

# ==================================================
# MLflow Tracking Server
# ==================================================

mlflow.set_tracking_uri("http://192.168.56.10:30500")

# ==================================================
# Load Latest Registered Model
# ==================================================

model = mlflow.pyfunc.load_model(
    "models:/iris-randomforest/latest"
)

# ==================================================
# FastAPI App
# ==================================================

app = FastAPI()


class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: IrisRequest):

    input_data = pd.DataFrame([{
        "sepal length (cm)": data.sepal_length,
        "sepal width (cm)": data.sepal_width,
        "petal length (cm)": data.petal_length,
        "petal width (cm)": data.petal_width
    }])

    prediction = model.predict(input_data)

    return {
        "prediction": int(prediction[0])
    }