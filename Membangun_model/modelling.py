import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor


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

mlflow.autolog()

with mlflow.start_run():

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

    print(
        f"R2 Score: {score}"
    )
