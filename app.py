from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/analyze')
def analyze():
    return render_template('analyze.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        tenure = float(request.form.get('tenure', 0))
        monthly = float(request.form.get('plan', 0))
        contract = int(request.form.get('contract', 0))
        internet = int(request.form.get('internet', 0))

        columns = [
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
        ]

        data = {col: 0 for col in columns}

        data['tenure'] = tenure
        data['MonthlyCharges'] = monthly
        data['TotalCharges'] = tenure * monthly

        if contract == 1:
            data['Contract_One year'] = 1
        elif contract == 2:
            data['Contract_Two year'] = 1

        if internet == 1:
            data['InternetService_Fiber optic'] = 1
        elif internet == 2:
            data['InternetService_No'] = 1

        df_input = pd.DataFrame([data])
        final_features = scaler.transform(df_input)
        prediction = model.predict(final_features)

        reasons_list = []
        suggestions_list = []

        if prediction[0] == 1:
            prediction_text = "High Risk"
            
            if tenure < 12:
                reasons_list.append("Low tenure (less than 12 months)")
                suggestions_list.append("Offer loyalty benefits and incentives")

            if monthly > 80:
                reasons_list.append("High monthly charges")
                suggestions_list.append("Provide better pricing options or discounts")

            if contract == 0:
                reasons_list.append("Month-to-month contract (no commitment)")
                suggestions_list.append("Encourage long-term contract with benefits")

            if internet == 1:
                reasons_list.append("Fiber optic users have higher churn rates")
                suggestions_list.append("Address service quality concerns")
            
            if not reasons_list:
                reasons_list.append("Multiple risk factors detected by model")
                suggestions_list.append("Schedule a customer retention call")
        else:
            prediction_text = "Low Risk"
            
            if tenure >= 12:
                reasons_list.append("Long tenure with company")

            if contract in [1, 2]:
                reasons_list.append("Long-term contract commitment")

            if monthly <= 50:
                reasons_list.append("Affordable monthly charges")

            if internet == 0:
                reasons_list.append("Stable DSL service")

            if not reasons_list:
                reasons_list.append("No major churn indicators")

        return render_template(
            'analyze.html',
            prediction_text=prediction_text,
            reasons_list=reasons_list,
            suggestions_list=suggestions_list
        )

    except Exception as e:
        return render_template('analyze.html', error=str(e))


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)