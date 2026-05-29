import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

FEATURE_COLUMNS = ["ScreenTime", "SocialMediaHours", "StudyHours", "SleepHours", "StressLevel"]
ENGINEERED_FEATURES = ["SocialPct", "StudySleepRatio", "ScreenMinusStudy", "StressPerHour", "DailyLoad"]
ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + ENGINEERED_FEATURES
TARGET_COLUMN = "RiskLevel"
ID_COLUMN = "ID"


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SocialPct"] = df["SocialMediaHours"] / df["ScreenTime"].replace(0, np.nan)
    df["SocialPct"] = df["SocialPct"].fillna(0.0)

    df["StudySleepRatio"] = df["StudyHours"] / df["SleepHours"].replace(0, np.nan)
    df["StudySleepRatio"] = df["StudySleepRatio"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    df["ScreenMinusStudy"] = df["ScreenTime"] - df["StudyHours"]
    df["StressPerHour"] = df["StressLevel"] / df["ScreenTime"].replace(0, np.nan)
    df["StressPerHour"] = df["StressPerHour"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    df["DailyLoad"] = df["ScreenTime"] + df["SocialMediaHours"] + df["StudyHours"]
    return df


RISK_SCORE_MAP = {
    "Low": 0.0,
    "Medium": 5.0,
    "High": 10.0,
}


def compute_risk_score(probabilities, class_names):
    if probabilities.ndim == 2:
        probabilities = probabilities[0]
    mapping = {}
    for index, label in enumerate(class_names):
        if label in RISK_SCORE_MAP:
            mapping[label] = RISK_SCORE_MAP[label]
        else:
            mapping[label] = float(index) * 10.0 / max(len(class_names) - 1, 1)

    score = sum(mapping[label] * float(probabilities[i]) for i, label in enumerate(class_names))
    return round(score, 2)


def build_advice(record: dict) -> str:
    advice = []
    if record.get("ScreenTime", 0) > 5:
        advice.append("reduce total screen time toward 5 hours or less")
    if record.get("SocialMediaHours", 0) > 2:
        advice.append("keep social media below 2 hours")
    if record.get("SleepHours", 0) < 7:
        advice.append("aim for at least 7 hours of sleep")
    if record.get("StudyHours", 0) < 2:
        advice.append("increase study time to 2+ hours")
    if record.get("StressLevel", 0) >= 4:
        advice.append("practice stress relief and regular breaks")
    if not advice:
        return "Your routine looks balanced; continue healthy habits."
    return "Try to " + "; ".join(advice) + "."


def get_model() -> object:
    return RandomForestClassifier(random_state=1)


def build_pipeline(model) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, digits=4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_artifact(obj, path: str) -> None:
    import joblib
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def save_label_encoder(encoder: LabelEncoder, path: str) -> None:
    import joblib
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, path)


def load_label_encoder(path: str) -> LabelEncoder:
    import joblib
    return joblib.load(path)


def encode_target(y: pd.Series) -> tuple[pd.Series, LabelEncoder]:
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder


def save_json(data, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(convert_numpy(data), handle, indent=2)


def save_processed_features(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_feature_importance(model, feature_names, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=feature_names)
        importance = importance.sort_values(ascending=False)
        importance.to_csv(path, header=["importance"])
    else:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("feature_importances not available for this estimator\n")
