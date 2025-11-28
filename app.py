from fastapi import FastAPI
from typing import List

from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn

app = FastAPI(title="Heart Disease Prediction API")

class HeartFeatures(BaseModel):
    Age: float
    Sex: str
    ChestPainType: str
    RestingBP: float
    Cholesterol: float
    FastingBS: float
    RestingECG: str
    MaxHR: float
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str

sex_map = {'M': 0, 'F': 1}
angina_map = {'N': 0, 'Y': 1}
cp_map = {'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3}
ecg_map = {'Normal': 0, 'ST': 1, 'LVH': 2}
slope_map = {'Up': 0, 'Flat': 1, 'Down': 2}


experiment_name = "heart_disease_classification_new"
experiment = mlflow.get_experiment_by_name(experiment_name)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"]
)


best_run_id  = runs.iloc[0]["run_id"]
print("Loaded model from run:", best_run_id)

model = mlflow.sklearn.load_model(
    f"runs:/{best_run_id }/model"
)
model_accuracy = runs.iloc[0]["metrics.accuracy"]
@app.post("/predict")
def predict(features : list[HeartFeatures]):
    processed_rows = []

    for item in features:
        row = item.dict()

        row["Sex"] = sex_map[row["Sex"]]
        row["ChestPainType"] = cp_map[row["ChestPainType"]]
        row["RestingECG"] = ecg_map[row["RestingECG"]]
        row["ExerciseAngina"] = angina_map[row["ExerciseAngina"]]
        row["ST_Slope"] = slope_map[row["ST_Slope"]] 

        processed_rows.append(row)
    df = pd.DataFrame(processed_rows)
    pred = model.predict(df)
    prob = model.predict_proba(df)[:, 1]

    results = []
    for p, pro in zip(pred,prob):
        results.append({
             "heart_disease_prediction": int(p),
             "probability": float(pro),
             "model_acc": float(model_accuracy)
        })

    return results
