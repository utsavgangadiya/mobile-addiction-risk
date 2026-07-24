# 🚀 Mobile Addiction Risk Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-black.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7.svg)](https://render.com/)
[![Live Demo](https://img.shields.io/badge/Live-Demo-success.svg)](https://mobile-addiction-risk.onrender.com/)

## 🌐 Live Demo

**🔗 https://mobile-addiction-risk.onrender.com/**

A production-ready end-to-end Machine Learning web application that predicts **Mobile Addiction Risk** using behavioral data and engineered features. The application is built with **Flask**, **scikit-learn**, and deployed on **Render** for real-time predictions.

---

# 📌 Features

- 🤖 End-to-End Machine Learning Pipeline
- 🌐 Live Flask Web Application
- 📊 Real-time Risk Prediction
- 🎯 Low / Medium / High Risk Classification
- 📈 Probability Score for Each Prediction
- ⚡ Feature Engineering Pipeline
- 🛡 Input Validation
- 📱 Responsive Dark-Themed UI
- 🔥 Production-ready Deployment on Render
- 🔗 REST API Support

---

# 🛠 Tech Stack

- Python
- Flask
- scikit-learn
- Pandas
- NumPy
- Joblib
- HTML5
- CSS3
- Render
- Git & GitHub

---

# 📂 Project Structure

```text
mobile-addiction-risk/
│
├── app.py
├── build_simple_model.py
├── model_utils.py
├── requirements.txt
├── README.md
│
├── artifacts/
│   ├── risk_pipeline.pkl
│   ├── label_encoder.pkl
│   ├── label_classes.json
│   ├── feature_columns.json
│   └── processed_features.csv
│
├── templates/
│   └── index.html
│
└── static/
    └── css/
        └── style.css
```

---

# 🚀 Live Deployment

The project is successfully deployed on **Render**.

### Live Website

https://mobile-addiction-risk.onrender.com/

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/utsavgangadiya/mobile-addiction-risk.git
```

Move into the project

```bash
cd mobile-addiction-risk
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open your browser

```
http://localhost:5000
```

---

# 🧠 Machine Learning Pipeline

The project follows a complete Machine Learning workflow:

- Data Preprocessing
- Feature Engineering
- Model Training
- Model Serialization
- Prediction Pipeline
- Flask Integration
- Live Deployment

The trained pipeline is stored as

```
artifacts/risk_pipeline.pkl
```

which is directly used for inference.

---

# 📊 Feature Engineering

The model generates additional informative features, including:

- Social Media Ratio
- Study Efficiency
- Screen Balance
- Stress Intensity
- Daily Activity Load

These engineered features improve prediction quality and model performance.

---

# 🎯 Prediction Output

The application predicts:

- Low Risk
- Medium Risk
- High Risk

Each prediction includes:

- Risk Category
- Risk Score
- Prediction Probability
- Personalized Recommendation

---

# 🔗 API Endpoint

### Predict Risk

```
POST /api/predict
```

Example Request

```json
{
    "ScreenTime": 5.2,
    "SocialMediaHours": 2.9,
    "StudyHours": 3.0,
    "SleepHours": 6.5,
    "StressLevel": 3
}
```

Example Response

```json
{
    "prediction": "Medium",
    "risk_score": 5.23,
    "probabilities": {
        "Low": 0.12,
        "Medium": 0.76,
        "High": 0.12
    },
    "advice": "Reduce daily screen time and improve sleep habits."
}
```


---

# ✨ Highlights

- End-to-End ML Project
- Flask Backend
- Production-ready Prediction Pipeline
- Real-time Predictions
- Responsive User Interface
- REST API
- Feature Engineering
- Live Deployment
- Clean Project Structure
- Portfolio-ready Project

---

# 🚀 Future Improvements

- User Authentication
- Prediction History
- Dashboard Analytics
- Explainable AI (SHAP)
- Docker Support
- Cloud Database
- Model Monitoring

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Commit your changes.
5. Open a Pull Request.

---

# 👨‍💻 Author

**Utsav Gangadiya**

GitHub: https://github.com/utsavgangadiya

Live Demo: https://mobile-addiction-risk.onrender.com/

---

## ⭐ Support

If you found this project useful, consider giving it a **Star ⭐** on GitHub.
