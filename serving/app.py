import os
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

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

model = mlflow.pyfunc.load_model("models:/iris-randomforest/latest")

class Request(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: Request):
    df = pd.DataFrame([{
        "sepal length (cm)": req.sepal_length,
        "sepal width (cm)": req.sepal_width,
        "petal length (cm)": req.petal_length,
        "petal width (cm)": req.petal_width
    }])

    pred = model.predict(df)
    return {"prediction": int(pred[0])}