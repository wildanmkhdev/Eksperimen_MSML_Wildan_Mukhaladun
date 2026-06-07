import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================================
# CONFIG MLFLOW
# =====================================================

mlflow.set_tracking_uri("file:./mlruns")

mlflow.set_experiment(
    "Student_Exam_Performance_Tuning"
)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(
    "student_exam_preprocessed.csv"
)

X = df.drop(
    "math score",
    axis=1
)

y = df["math score"]

# =====================================================
# SPLIT DATA
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# HYPERPARAMETER TUNING
# =====================================================

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5]
}

rf = RandomForestRegressor(
    random_state=42
)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(
    X_train,
    y_train
)

best_model = grid_search.best_estimator_

# =====================================================
# PREDICTION
# =====================================================

y_pred = best_model.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)

# =====================================================
# MANUAL LOGGING
# =====================================================

with mlflow.start_run():

    # log parameter terbaik

    mlflow.log_param(
        "n_estimators",
        grid_search.best_params_["n_estimators"]
    )

    mlflow.log_param(
        "max_depth",
        grid_search.best_params_["max_depth"]
    )

    mlflow.log_param(
        "min_samples_split",
        grid_search.best_params_["min_samples_split"]
    )

    # log metrics

    mlflow.log_metric(
        "MAE",
        mae
    )

    mlflow.log_metric(
        "MSE",
        mse
    )

    mlflow.log_metric(
        "RMSE",
        rmse
    )

    mlflow.log_metric(
        "R2",
        r2
    )

    # simpan model

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best_model"
    )

    print("=" * 50)
    print("BEST PARAMETER")
    print(grid_search.best_params_)
    print("=" * 50)

    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")