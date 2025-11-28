import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score , classification_report ,f1_score


import mlflow
import mlflow.sklearn

df = pd.read_csv("data/heart.csv")

df['Sex'] = df['Sex'].map({'M': 0, 'F': 1})
df['ExerciseAngina'] = df['ExerciseAngina'].map({'N': 0, 'Y': 1})
df['ChestPainType'] = df['ChestPainType'].map({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
df['RestingECG'] = df['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
df['ST_Slope'] = df['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})

X = df.drop("HeartDisease", axis=1).astype(float)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("heart_disease_classification_new")

models ={
        "Logistic_Regression" : LogisticRegression(max_iter=300),
         "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
         "svm": SVC( kernel="rbf",C=1.0,gamma="scale",)
        }

best_model = None
best_score = -1
best_model_name = None

for model_name,model in models.items():
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test , pred)
        f1 = f1_score(y_test , pred)

        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            model,
            artifact_path='model',
            input_example=X_train.head(3)
        )

        print(f"model : {model_name} | Accuracy : {acc:.4f} | F1 :{f1:.4f}")

        if acc > best_score:
            best_score = acc
            best_model = model
            best_model_name = model_name
            best_run_id = mlflow.active_run().info.run_id

print(f" Best Model: {best_model_name}")
print(f" Best Accuracy: {best_score}")
print(f" Best Run ID: {best_run_id}")
    