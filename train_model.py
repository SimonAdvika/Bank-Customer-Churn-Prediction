import os
import pickle

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)




DATA_PATH = "European_Bank.csv"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "bank_churn_mlp.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "churn_scaler.pkl"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "feature_columns.pkl"
)


# Create models folder
os.makedirs(MODEL_DIR, exist_ok=True)



print("\n" + "=" * 60)
print("BANK CUSTOMER CHURN  MODEL TRAINING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])



print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())




possible_targets = [
    "Exited",
    "Churn",
    "churn",
    "Customer_Churn",
    "CustomerChurn"
]

target_column = None

for column in possible_targets:

    if column in df.columns:
        target_column = column
        break


if target_column is None:

    raise ValueError(
        "\nTarget column not found.\n"
        "Expected one of:\n"
        "Exited, Churn, churn, Customer_Churn, CustomerChurn"
    )


print("\nTarget column:", target_column)



columns_to_drop = [
    "RowNumber",
    "CustomerId",
    "CustomerID",
    "Surname",
    "Year"
]

columns_to_drop = [
    column
    for column in columns_to_drop
    if column in df.columns
]


if columns_to_drop:

    print(
        "\nRemoving unnecessary columns:",
        columns_to_drop
    )

    df = df.drop(
        columns=columns_to_drop
    )



print("\nChecking missing values...")

missing_values = df.isnull().sum().sum()

print(
    "Total missing values:",
    missing_values
)


if missing_values > 0:

    print(
        "\nMissing values detected."
        " Removing rows with missing values..."
    )

    df = df.dropna()




X = df.drop(
    columns=[target_column]
)

y = df[target_column]




if y.dtype == "object":

    y = (
        y.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "churn": 1,
            "retained": 0
        })
    )


y = pd.to_numeric(
    y,
    errors="coerce"
)


valid_rows = y.notna()

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()


y = y.astype(int)



print("\nTarget distribution:")

print(
    y.value_counts()
)




print("\nEncoding categorical columns...")

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()


print(
    "Categorical columns:",
    categorical_columns
)


X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=False
)



for column in X.columns:

    if X[column].dtype == bool:

        X[column] = X[column].astype(int)



feature_columns = X.columns.tolist()


print("\nFinal model features:")

for i, column in enumerate(
    feature_columns,
    start=1
):

    print(
        f"{i:02d}. {column}"
    )


with open(
    FEATURE_PATH,
    "wb"
) as file:

    pickle.dump(
        feature_columns,
        file
    )


print(
    "\nFeature columns saved:",
    FEATURE_PATH
)



X = X.astype(float)

X_values = X.values

y_values = y.values




X_train, X_test, y_train, y_test = train_test_split(

    X_values,

    y_values,

    test_size=0.20,

    random_state=42,

    stratify=y_values
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))



print("\nScaling numerical features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)


# Save scaler
with open(
    SCALER_PATH,
    "wb"
) as file:

    pickle.dump(
        scaler,
        file
    )


print(
    "Scaler saved:",
    SCALER_PATH
)



print("\nBuilding neural network (scikitlearn MLP, no TensorFlow needed)...")

model = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    batch_size=32,
    alpha=1e-4,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.20,
    n_iter_no_change=10,
    random_state=42,
    verbose=True
)

print("\nStarting training...")
model.fit(X_train, y_train)

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

print("\nNeural network model saved:", MODEL_PATH)


print("\nEvaluating model...")

accuracy = model.score(X_test, y_test)
print(f"\nTest Accuracy : {accuracy:.4f}")
print(f"Accuracy      : {accuracy * 100:.2f}%")


probabilities = model.predict_proba(X_test)[:, 1]


predictions = (
    probabilities >= 0.5
).astype(int)




print("\n" + "=" * 60)

print("CLASSIFICATION REPORT")

print("=" * 60)

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Retained",
            "Churn"
        ]
    )
)



print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)




print("\n" + "=" * 60)

print("TRAINING COMPLETED")
print("=" * 60)

print("\nCreated files:")

print(
    "1.",
    MODEL_PATH
)

print(
    "2.",
    SCALER_PATH
)

print(
    "3.",
    FEATURE_PATH
)

print("\nYour project is ready for Flask prediction.")