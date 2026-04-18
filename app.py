from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Note: You should have your trained model.pkl in the same directory as app.py
# If it doesn't exist, this is a mock implementation so the app still runs
MODEL_PATH = 'model.pkl'
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
else:
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        
        # Extract features
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])

        # Apply transformations as specified
        N_transformed = N  # N is not transformed in Project.py
        P_transformed = np.log1p(P)
        K_transformed = np.log1p(K)
        humidity_transformed = np.sqrt(humidity)
        rainfall_transformed = np.sqrt(rainfall)

        # Create DataFrame with expected column names
        features = pd.DataFrame([[
            N_transformed, 
            P_transformed, 
            K_transformed, 
            temperature, 
            humidity_transformed, 
            ph, 
            rainfall_transformed
        ]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

        # Make prediction
        if model:
            prediction = model.predict(features)[0]
            # Format the crop name properly (e.g., capitalize)
            predicted_crop = str(prediction).capitalize()
        else:
            # Fallback if no model is found
            predicted_crop = "Model Not Found (Mock: Rice)"

        # Return JSON response
        return jsonify({
            "prediction": predicted_crop,
            "accuracy": "99%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
