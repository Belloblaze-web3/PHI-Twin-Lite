from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from generate_data import FAULTS, build_dataset

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "landing_gear_data.csv"
MODEL_PATH = ROOT / "model.pkl"


def main():
    df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else build_dataset(output_path=str(DATA_PATH))
    feature_columns = [column for column in df.columns if column not in ("label", "fault_name")]
    X = df[feature_columns]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    labels = sorted(y.unique())
    names = [FAULTS.get(int(label), str(label)) for label in labels]
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print(classification_report(y_test, predictions, labels=labels, target_names=names, zero_division=0))
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
