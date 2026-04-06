from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        tenure = float(request.form['tenure'])
        monthly = float(request.form['monthly'])
        contract = int(request.form['contract'])
        internet = int(request.form['internet'])

        # Create feature dictionary (same as training columns)
        data = {col: 0 for col in [
            'gender','SeniorCitizen','tenure','MonthlyCharges','TotalCharges',
            'Partner_Yes','Dependents_Yes','PhoneService_Yes',
            'MultipleLines_No phone service','MultipleLines_Yes',
            'InternetService_Fiber optic','InternetService_No',
            'OnlineSecurity_No internet service','OnlineSecurity_Yes',
            'OnlineBackup_No internet service','OnlineBackup_Yes',
            'DeviceProtection_No internet service','DeviceProtection_Yes',
            'TechSupport_No internet service','TechSupport_Yes',
            'StreamingTV_No internet service','StreamingTV_Yes',
            'StreamingMovies_No internet service','StreamingMovies_Yes',
            'Contract_One year','Contract_Two year','PaperlessBilling_Yes',
            'PaymentMethod_Credit card (automatic)',
            'PaymentMethod_Electronic check','PaymentMethod_Mailed check'
        ]}

        # Fill important fields
        data['tenure'] = tenure
        data['MonthlyCharges'] = monthly
        data['TotalCharges'] = tenure * monthly  # approximation

        # Contract encoding
        if contract == 1:
            data['Contract_One year'] = 1
        elif contract == 2:
            data['Contract_Two year'] = 1

        # Internet encoding
        if internet == 1:
            data['InternetService_Fiber optic'] = 1
        elif internet == 2:
            data['InternetService_No'] = 1

        # Convert to DataFrame
        df_input = pd.DataFrame([data])

        # Scale input
        final_features = scaler.transform(df_input)

        # Prediction
        prediction = model.predict(final_features)

        if prediction[0] == 1:
            result = "⚠️ Customer is likely to churn"
        else:
            result = "✅ Customer is not likely to churn"

        # -------------------------
        # Rule-based explanation
        # -------------------------
        reasons = []

        if tenure < 12:
            reasons.append("Low tenure (new customer)")

        if monthly > 80:
            reasons.append("High monthly charges")

        if contract == 0:
            reasons.append("Month-to-month contract (no long-term commitment)")

        if internet == 1:
            reasons.append("Fiber optic users tend to churn more")

        if not reasons:
            reason_text = "No major risk factors detected"
        else:
            reason_text = "Reasons: " + ", ".join(reasons)

        return render_template('index.html',
                               prediction_text=result,
                               reasons=reason_text)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)