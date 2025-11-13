from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = float(request.form['age'])
        
        # Prepare input for prediction
        input_features = np.array([[pregnancies, glucose, blood_pressure, 
                                    skin_thickness, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_features)
        prediction = model.predict(input_scaled)[0]

        # Determine the result
        result = 'Diabetes Detected' if prediction == 1 else 'No Diabetes'

        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
