import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from model_utils import (
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_features,
    build_pipeline,
    encode_target,
    get_model,
    save_artifact,
    save_json,
    save_label_encoder,
    save_processed_features,
)


def generate_synthetic_data(n_samples: int = 300, random_state: int = 1) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)
    df = pd.DataFrame(
        {
            "ScreenTime": rng.uniform(1.0, 10.0, size=n_samples).round(2),
            "SocialMediaHours": rng.uniform(0.0, 5.0, size=n_samples).round(2),
            "StudyHours": rng.uniform(0.0, 5.0, size=n_samples).round(2),
            "SleepHours": rng.uniform(4.0, 9.0, size=n_samples).round(2),
            "StressLevel": rng.randint(1, 6, size=n_samples),
        }
    )

    score = (
        0.35 * (df["ScreenTime"] / 10.0)
        + 0.25 * (df["SocialMediaHours"] / 5.0)
        + 0.20 * ((5.0 - df["StudyHours"]) / 5.0)
        + 0.10 * ((7.0 - df["SleepHours"]).clip(lower=0.0) / 3.0)
        + 0.10 * ((df["StressLevel"] - 1) / 4.0)
    )

    labels = []
    for value in score:
        if value >= 0.65:
            labels.append("High")
        elif value >= 0.35:
            labels.append("Medium")
        else:
            labels.append("Low")

    df[TARGET_COLUMN] = labels
    return df


def main(args):
    artifact_dir = Path(args.output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_data(n_samples=args.samples, random_state=args.random_state)
    df = build_features(df)

    X = df[ALL_FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    y_encoded, label_encoder = encode_target(y)

    pipeline = build_pipeline(get_model())
    pipeline.fit(X, y_encoded)

    save_artifact(pipeline, artifact_dir / "risk_pipeline.pkl")
    save_label_encoder(label_encoder, artifact_dir / "label_encoder.pkl")
    save_json(label_encoder.classes_.tolist(), artifact_dir / "label_classes.json")
    save_json(ALL_FEATURE_COLUMNS, artifact_dir / "feature_columns.json")
    save_processed_features(df[[*ALL_FEATURE_COLUMNS, TARGET_COLUMN]], artifact_dir / "processed_features.csv")

    print("Simple model training complete.")
    print(f"Saved pipeline to: {artifact_dir / 'risk_pipeline.pkl'}")
    print(f"Saved label encoder to: {artifact_dir / 'label_encoder.pkl'}")
    print(f"Saved processed dataset to: {artifact_dir / 'processed_features.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a simple deployment-ready model from synthetic data.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory to save model artifacts.")
    parser.add_argument("--samples", type=int, default=300, help="Number of synthetic records to generate.")
    parser.add_argument("--random-state", type=int, default=1, help="Random seed for synthetic data generation.")
    args = parser.parse_args()
    main(args)
