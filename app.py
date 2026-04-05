from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        tenure = float(request.form['tenure'])
        monthly = float(request.form['monthly'])
        contract = int(request.form['contract'])
        internet = int(request.form['internet'])

        # create input array (simple version)
        features = [tenure, monthly, contract, internet]

        # ⚠️ Temporary fix (pad to 30 features)
        features = features + [0]*(30 - len(features))

        final_features = np.array([features])
        final_features = scaler.transform(final_features)

        prediction = model.predict(final_features)

        if prediction[0] == 1:
            result = " ⚠️Customer is likely to churn"
        else:
            result = " ✅Customer is not likely to churn"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)