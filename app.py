import numpy as np
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

numerical_indices = [1, 4, 17, 18]  
# SeniorCitizen, tenure, MonthlyCharges, TotalCharges

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    features = [
        int(request.form['gender']),
        int(request.form['SeniorCitizen']),
        int(request.form['Partner']),
        int(request.form['Dependents']),
        float(request.form['tenure']),
        int(request.form['PhoneService']),
        int(request.form['MultipleLines']),
        int(request.form['InternetService']),
        int(request.form['OnlineSecurity']),
        int(request.form['OnlineBackup']),
        int(request.form['DeviceProtection']),
        int(request.form['TechSupport']),
        int(request.form['StreamingTV']),
        int(request.form['StreamingMovies']),
        int(request.form['Contract']),
        int(request.form['PaperlessBilling']),
        int(request.form['PaymentMethod']),
        float(request.form['MonthlyCharges']),
        float(request.form['TotalCharges'])
    ]

    X = np.array(features).reshape(1, -1)

    # ✅ scale ONLY numeric features
    X[:, numerical_indices] = scaler.transform(X[:, numerical_indices])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

# churn yes/no
    churn_status = "Yes" if prediction == 1 else "No"

# risk level
    if probability >= 0.7:
        risk_level = "High"
    elif probability >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return render_template(
    "index.html",
    churn_status=churn_status,
    risk_level=risk_level
)


if __name__ == "__main__":
    app.run(debug=True)
