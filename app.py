import json
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

from model_utils import (
    ALL_FEATURE_COLUMNS,
    build_features,
    load_label_encoder,
    compute_risk_score,
    build_advice,
)

MODEL_PATH = Path("artifacts/risk_pipeline.pkl")

app = Flask(__name__, static_folder="static", template_folder="templates")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first.")
    return joblib.load(MODEL_PATH)


pipeline = load_model()
label_encoder = None
label_encoder_path = Path("artifacts/label_encoder.pkl")
if label_encoder_path.exists():
    label_encoder = load_label_encoder(label_encoder_path)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.form.to_dict()
    record = {}
    errors = []
    for feature in ALL_FEATURE_COLUMNS[:5]:
        value = payload.get(feature)
        if value is None or value.strip() == "":
            errors.append(f"{feature} is required.")
            continue
        try:
            record[feature] = float(value)
        except ValueError:
            errors.append(f"{feature} must be a number.")

    if errors:
        return render_template("index.html", errors=errors, values=payload)

    import pandas as pd
    df = pd.DataFrame([record])
    df = build_features(df)
    prediction = pipeline.predict(df[ALL_FEATURE_COLUMNS])
    probabilities = pipeline.predict_proba(df[ALL_FEATURE_COLUMNS])[0]
    if label_encoder is not None:
        class_names = label_encoder.classes_
        prediction = label_encoder.inverse_transform(prediction)
    else:
        class_names = pipeline.classes_
    risk_score = compute_risk_score(probabilities, class_names)
    advice = build_advice(record)
    prob_map = {str(class_names[i]): round(float(probabilities[i]), 4) for i in range(len(class_names))}
    return render_template(
        "index.html",
        prediction=str(prediction[0]),
        probability=prob_map,
        risk_score=risk_score,
        advice=advice,
        values=payload,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON payload."}), 400

    if not isinstance(payload, dict):
        return jsonify({"error": "JSON body must be an object."}), 400

    record = {}
    for feature in ALL_FEATURE_COLUMNS[:5]:
        value = payload.get(feature)
        if value is None:
            return jsonify({"error": f"Missing field '{feature}'."}), 400
        try:
            record[feature] = float(value)
        except ValueError:
            return jsonify({"error": f"Field '{feature}' must be numeric."}), 400

    import pandas as pd
    df = pd.DataFrame([record])
    df = build_features(df)
    prediction = pipeline.predict(df[ALL_FEATURE_COLUMNS])[0]
    probabilities = pipeline.predict_proba(df[ALL_FEATURE_COLUMNS])[0]
    if label_encoder is not None:
        class_names = label_encoder.classes_
        prediction = label_encoder.inverse_transform([prediction])[0]
    else:
        class_names = pipeline.classes_
    risk_score = compute_risk_score(probabilities, class_names)
    advice = build_advice(record)
    return jsonify({
        "prediction": str(prediction),
        "risk_score": risk_score,
        "advice": advice,
        "probabilities": {str(class_names[i]): float(probabilities[i]) for i in range(len(probabilities))},
        "features": record,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)