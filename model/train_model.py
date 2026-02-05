import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "results/features.csv"
MODEL_PATH = "results/model.pkl"

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    if df.empty:
        print("Dataset is empty. Cannot train model.")
        return

    # Separate features and label
    X = df.drop(columns=["label"])
    y = df["label"]

    # Convert non-numeric columns to numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Model training completed")
    print("Accuracy:", acc)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, MODEL_PATH)
    print("Model saved to", MODEL_PATH)

if __name__ == "__main__":
    main()
