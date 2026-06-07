import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

INPUT_FILE = "../dataset/student_exam_performance.csv"
OUTPUT_FILE = "student_exam_preprocessed.csv"


def preprocess():

    df = pd.read_csv("student_exam_performance.csv")

    target = "math score"

    X = df.drop(columns=[target])

    y = df[target]

    categorical_features = [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course"
    ]

    numerical_features = [
        "reading score",
        "writing score"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "num",
                StandardScaler(),
                numerical_features
            )
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    encoded_columns = (
        preprocessor
        .named_transformers_["cat"]
        .get_feature_names_out(categorical_features)
    )

    all_columns = list(encoded_columns) + numerical_features
    processed_df = pd.DataFrame(
    X_processed,
    columns=all_columns
)

    processed_df[target] = y.values

    processed_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("Preprocessing selesai")
    print(processed_df.shape)


if __name__ == "__main__":
    preprocess()