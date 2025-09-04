from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import os

app = Flask(__name__)
CORS(app)

# Load the pickled model
try:
    with open('xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: xgboost_model.pkl not found. Make sure the model file is in the same directory.")
    model = None

# Define the feature names (based on your preprocessing)
FEATURE_NAMES = [
    'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
    'Sex_F', 'Sex_M',
    'ChestPainType_ASY', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
    'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST',
    'ExerciseAngina_N', 'ExerciseAngina_Y',
    'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up'
]

def preprocess_input(data):
    """
    Preprocess the input data to match the model's expected format
    """
    # Create a dataframe with all features initialized to 0
    processed_data = pd.DataFrame(0, index=[0], columns=FEATURE_NAMES)
    
    # Set numerical features
    processed_data['Age'] = data['age']
    processed_data['RestingBP'] = data['resting_bp']
    processed_data['Cholesterol'] = data['cholesterol']
    processed_data['FastingBS'] = 1 if data['fasting_bs'] else 0
    processed_data['MaxHR'] = data['max_hr']
    processed_data['Oldpeak'] = data['oldpeak']
    
    # Set categorical features using one-hot encoding
    processed_data[f"Sex_{data['sex']}"] = 1
    processed_data[f"ChestPainType_{data['chest_pain_type']}"] = 1
    processed_data[f"RestingECG_{data['resting_ecg']}"] = 1
    processed_data[f"ExerciseAngina_{data['exercise_angina']}"] = 1
    processed_data[f"ST_Slope_{data['st_slope']}"] = 1
    
    return processed_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get data from request
        data = request.json
        
        # Validate required fields
        required_fields = [
            'age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol',
            'fasting_bs', 'resting_ecg', 'max_hr', 'exercise_angina', 'oldpeak', 'st_slope'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Preprocess the input
        processed_data = preprocess_input(data)
        
        # Convert to DMatrix for XGBoost
        dmatrix = xgb.DMatrix(processed_data)
        
        # Make prediction
        prediction_proba = model.predict(dmatrix)[0]
        prediction = int(prediction_proba > 0.5)
        
        # Prepare response
        response = {
            'prediction': prediction,
            'probability': float(prediction_proba),
            'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
            'confidence': float(max(prediction_proba, 1 - prediction_proba))
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    app.run(debug=True, host='0.0.0.0', port=5000)