# prediction.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from flask import jsonify
import json

class PricePrediction:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.product_history = {
            'GOLDSTITCH Men Vintage Tailcoat': {
                'prices': [68, 70, 71, 69, 72, 70, 68, 71, 73, 70],
                'id': 'f1'
            },
            'Cartoon Astronaut T-Shirts': {
                'prices': [78, 80, 82, 79, 81, 78, 80, 82, 78, 81],
                'id': 'f4'
            },
            'HP Victus Gaming Laptop': {
                'prices': [100, 98, 102, 99, 101, 103, 100, 98, 102, 100],
                'id': 'f7'
            },
            'Oneplus Nord CE4': {
                'prices': [78, 80, 79, 81, 77, 79, 80, 78, 81, 79],
                'id': 'n1'
            },
            'Fire-Boltt Ninja Call Pro Max': {
                'prices': [80, 78, 82, 79, 81, 80, 78, 81, 79, 80],
                'id': 'n4'
            },
            'Wellcore Creatine': {
                'prices': [11, 10, 12, 11, 10, 11, 12, 10, 11, 12],
                'id': 'n7'
            }
        }
        self._initialize_models()

    def _initialize_models(self):
        for product_name in self.product_history.keys():
            self._create_model(product_name)

    def _create_model(self, product_name):
        prices = self.product_history[product_name]['prices']
        
        # Create and fit scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(np.array(prices).reshape(-1, 1))
        
        # Prepare sequences
        sequence_length = 5
        x_train, y_train = [], []
        
        for i in range(sequence_length, len(scaled_data)):
            x_train.append(scaled_data[i-sequence_length:i, 0])
            y_train.append(scaled_data[i, 0])
        
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        
        # Create and train model
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)),
            LSTM(units=50),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)
        
        self.models[product_name] = model
        self.scalers[product_name] = scaler

    def predict_price(self, product_name):
        if product_name not in self.product_history:
            return {
                'status': 'error',
                'error': 'Product not found'
            }

        try:
            prices = self.product_history[product_name]['prices']
            scaler = self.scalers[product_name]
            model = self.models[product_name]
            
            # Prepare last 5 prices for prediction
            last_sequence = np.array(prices[-5:])
            scaled_sequence = scaler.transform(last_sequence.reshape(-1, 1))
            scaled_sequence = scaled_sequence.reshape(1, 5, 1)
            
            # Make prediction
            scaled_prediction = model.predict(scaled_sequence)
            prediction = scaler.inverse_transform(scaled_prediction)[0][0]
            
            # Get price trend
            current_price = prices[-1]
            trend = "stable"
            if prediction > current_price * 1.05:
                trend = "increasing"
            elif prediction < current_price * 0.95:
                trend = "decreasing"
            
            return {
                'status': 'success',
                'product': product_name,
                'current_price': current_price,
                'predicted_price': round(prediction, 2),
                'trend': trend,
                'history': prices[-5:],
                'product_id': self.product_history[product_name]['id']
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

# Initialize the predictor
price_predictor = PricePrediction()

def predict_price(product_name):
    return jsonify(price_predictor.predict_price(product_name))