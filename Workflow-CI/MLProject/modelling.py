import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import mlflow
import mlflow.sklearn

mlflow.set_experiment("Workflow_CI_Training")

df = pd.read_csv(
    "student_exam_preprocessed.csv"
)

X = df.drop(
    "math score",
    axis=1
)

y = df["math score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    random_state=42
)

model.fit(
    X_train,
    y_train
)

score = model.score(
    X_test,
    y_test
)

mlflow.log_metric(
    "R2",
    score
)

mlflow.sklearn.log_model(
    model,
    "model"
)

print(
    f"R2 = {score}"
)
