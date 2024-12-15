# Flask API to use the trained model
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler

# Load the saved model and preprocessing components
model = tf.keras.models.load_model('model2.h5')
label_encoder_classes = np.load('label_encoder_classes.npy', allow_pickle=True)
scaler_mean = np.load('scaler_params.npy')
scaler_scale = np.load('scaler_scale.npy')
scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale

# Reinitialize the LabelEncoder with saved classes
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.classes_ = label_encoder_classes

# Load drug information dataset
drugs_df = pd.read_csv('drugs.csv')
drugs_df['Disease name'] = drugs_df['Disease name'].str.strip().str.lower()

# Flask API initialization
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON request
        data = request.get_json()

        # Validate input
        if 'symptoms' not in data:
            return jsonify({'error': 'Symptoms data is missing in the request body.'}), 400

        input_symptoms = np.array(data['symptoms'])

        # Ensure the input is binary (0 or 1)
        if not np.all((input_symptoms == 0) | (input_symptoms == 1)):
            return jsonify({'error': 'All input features must be binary (0 or 1).'}), 400

        # Reshape and scale the input
        input_symptoms = input_symptoms.reshape(1, -1)
        if input_symptoms.shape[1] != scaler.mean_.shape[0]:
            return jsonify({'error': f'Input should have {scaler.mean_.shape[0]} features.'}), 400

        input_symptoms_scaled = scaler.transform(input_symptoms)

        # Prediction
        predictions = model.predict(input_symptoms_scaled)
        predicted_index = np.argmax(predictions)
        predicted_class = le.inverse_transform([predicted_index])[0]

        # Confidence score converted to percentage
        confidence = float(predictions[0][predicted_index]) * 100

        # Normalize predicted class for comparison
        predicted_class_normalized = predicted_class.strip().lower()

        # Fetch drug info
        drug_info = drugs_df[drugs_df['Disease name'] == predicted_class_normalized]
        if drug_info.empty:
            return jsonify({
                'disease': predicted_class,
                'confidence': round(confidence, 2),
                'drug_info': None
            })

        # Extracting relevant drug data and replacing NaN with null
        drug_details = drug_info[['Drug', 'Side effect and allergy', 'PT', 'exercise and link', 'Nutrition']].where(pd.notnull(drug_info), None).to_dict(orient='records')

        return jsonify({
            'disease': predicted_class,
            'confidence': round(confidence, 2),
            'drug_info': drug_details
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500



