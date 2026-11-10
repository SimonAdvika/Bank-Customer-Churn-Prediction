import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "bank_churn_mlp.pkl"
SCALER_PATH = BASE_DIR / "models" / "churn_scaler.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_columns.pkl"

CATEGORIES = {
    "Geography": ["France", "Germany", "Spain"],
    "Gender": ["Female", "Male"],
}

for path in (MODEL_PATH, SCALER_PATH, FEATURE_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}\nRun `python train_model.py` first.")


def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


print("Loading model...")
model = load_pickle(MODEL_PATH)
scaler = load_pickle(SCALER_PATH)
feature_columns = list(load_pickle(FEATURE_PATH))

expected = model.n_features_in_
if len(feature_columns) != expected or scaler.n_features_in_ != expected:
    raise RuntimeError(
        f"Model expects {expected} features but feature_columns has "
        f"{len(feature_columns)} and scaler has {scaler.n_features_in_}.\n"
        "Run `python train_model.py` to regenerate all three files together."
    )

print("Model, scaler and features loaded. Features:", feature_columns)


def read_number(data, key, default, low, high):
    value = float(data.get(key, default))
    if not np.isfinite(value) or not (low <= value <= high):
        raise ValueError(f"'{key}' must be between {low} and {high}.")
    return value


def build_features(customer):
    """One-hot encode exactly like train_model.py (pd.get_dummies, drop_first=False)."""
    encoded = pd.get_dummies(customer, columns=["Geography", "Gender"], drop_first=False)
    # Columns absent from this single row (other categories) are filled with 0.
    return encoded.reindex(columns=feature_columns, fill_value=0).astype(float)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"success": False, "error": "No JSON data received."}), 400

        geography = str(data.get("geography", "France"))
        gender = str(data.get("gender", "Male"))
        if geography not in CATEGORIES["Geography"]:
            raise ValueError(f"Unknown geography '{geography}'.")
        if gender not in CATEGORIES["Gender"]:
            raise ValueError(f"Unknown gender '{gender}'.")

        customer = pd.DataFrame([{
            "CreditScore": read_number(data, "credit_score", 650, 300, 900),
            "Geography": geography,
            "Gender": gender,
            "Age": read_number(data, "age", 35, 18, 100),
            "Tenure": read_number(data, "tenure", 5, 0, 50),
            "Balance": read_number(data, "balance", 50000, 0, 1e9),
            "NumOfProducts": read_number(data, "products", 1, 1, 10),
            "HasCrCard": int(read_number(data, "has_card", 1, 0, 1)),
            "IsActiveMember": int(read_number(data, "active_member", 1, 0, 1)),
            "EstimatedSalary": read_number(data, "salary", 50000, 0, 1e9),
        }])

        features = build_features(customer)
        scaled = scaler.transform(features.values)

        probability = float(model.predict_proba(scaled)[0, 1])
        probability = min(max(probability, 0.0), 1.0)

        churn = probability >= 0.50
        return jsonify({
            "success": True,
            "prediction": "Customer is likely to CHURN" if churn else "Customer is likely to STAY",
            "status": "churn" if churn else "stay",
            "probability": round(probability * 100, 2),
        })

    except ValueError as error:
        return jsonify({"success": False, "error": str(error)}), 400
    except Exception as error:
        print("Prediction error:", error)
        return jsonify({"success": False, "error": str(error)}), 500


if __name__ == "__main__":
    print("\nBANK CUSTOMER CHURN PREDICTION")
    print("Open: http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)