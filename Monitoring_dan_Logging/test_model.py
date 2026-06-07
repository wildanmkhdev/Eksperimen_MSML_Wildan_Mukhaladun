import joblib

model = joblib.load(
    "model/model.pkl"
)

print(type(model))

print("MODEL BERHASIL DILOAD")
