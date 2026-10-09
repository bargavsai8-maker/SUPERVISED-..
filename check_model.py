import joblib

model = joblib.load("breast_cancer_model (3).pkl")

print("Model type:", type(model).name)
print("Number of features:", getattr(model, "n_features_in_", "Unknown"))

if hasattr(model, "classes_"):
print("Classes:", model.classes_)