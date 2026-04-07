from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # -------------------------
        # SAFE INPUT HANDLING
        # -------------------------
        tenure = float(request.form.get('tenure', 0))
        monthly = float(request.form.get('plan', 0))
        contract = int(request.form.get('contract', 0))
        internet = int(request.form.get('internet', 0))

        print("FORM DATA:", request.form)  # debug

        # -------------------------
        # CREATE FEATURE STRUCTURE
        # -------------------------
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

        # -------------------------
        # FILL IMPORTANT VALUES
        # -------------------------
        data['tenure'] = tenure
        data['MonthlyCharges'] = monthly
        data['TotalCharges'] = tenure * monthly

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

        df_input = pd.DataFrame([data])

        # -------------------------
        # SCALE + PREDICT
        # -------------------------
        final_features = scaler.transform(df_input)
        prediction = model.predict(final_features)

        print("Prediction:", prediction[0])  # debug

        # -------------------------
        # RESULT
        # -------------------------
        if prediction[0] == 1:
            result = "⚠️ High Risk: Customer is likely to churn"
        else:
            result = "✅ Low Risk: Customer is likely to stay"

        # -------------------------
        # REASONS
        # -------------------------
        reasons = []

        if tenure < 12:
            reasons.append("Low tenure")

        if monthly > 80:
            reasons.append("High monthly charges")

        if contract == 0:
            reasons.append("Month-to-month contract")

        if internet == 1:
            reasons.append("Fiber users churn more")

        reason_text = "Reasons: " + ", ".join(reasons) if reasons else "No major risk factors"

        # -------------------------
        # SUGGESTIONS
        # -------------------------
        suggestions = []

        if tenure < 12:
            suggestions.append("Offer loyalty benefits")

        if contract == 0:
            suggestions.append("Encourage long-term plan")

        if monthly > 80:
            suggestions.append("Provide better pricing")

        suggestion_text = "Suggestions: " + ", ".join(suggestions) if suggestions else "Customer looks stable"

        return render_template(
            'index.html',
            prediction_text=result,
            reasons=reason_text,
            suggestions=suggestion_text
        )

    except Exception as e:
        return str(e)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/analyze')
def analyze():
    return render_template('analyze.html')

if __name__ == "__main__":
    app.run(debug=True)