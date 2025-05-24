from flask import Flask, request, jsonify
import pickle
import numpy as np
import logging

logging.info('first - before any code')

app = Flask(__name__)
# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

logging.info('after pickle load')

@app.route('/')
def home():
    return jsonify({'message': 'Sales Prediction API is running'})

logging.info('after home')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'predicted_sales': prediction[0]})

logging.info('Logging statement before app.run - before if')
if __name__ == '__main__':
    logging.info('Logging statement before app.run - inside if')
    app.run(host='0.0.0.0', port=8080)
