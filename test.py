import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

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

run_id ="8906296669d0413ea3e60d193df60df7"
model  = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

pred = model.predict(X_test)


acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred)

print("✅ Accuracy on test data:", acc)
print("✅ F1 Score on test data:", f1)