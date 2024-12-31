import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class PricePrediction:
    def __init__(self, csv_file):
        self.models = {}
        self.scalers = {}
        self.product_history = self._load_product_history(csv_file)
        self._initialize_models()

    def _load_product_history(self, csv_file):
        product_history = {}
        try:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                product_history[row['Product Name']] = {
                    'prices': list(map(float, row['Prices'].split(','))),
                    'id': row['ID']
                }
        except Exception as e:
            print(f"Error loading product history: {e}")
        return product_history

    def _initialize_models(self):
        for product_name in self.product_history.keys():
            self._create_model(product_name)

    def _create_model(self, product_name):
        prices = self.product_history[product_name]['prices']
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(np.array(prices).reshape(-1, 1))
        
        sequence_length = 5
        x_train, y_train = [], []
        for i in range(sequence_length, len(scaled_data)):
            x_train.append(scaled_data[i-sequence_length:i, 0])
            y_train.append(scaled_data[i, 0])
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        
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
            return {'status': 'error', 'error': 'Product not found'}

        try:
            prices = self.product_history[product_name]['prices']
            scaler = self.scalers[product_name]
            model = self.models[product_name]
            
            last_sequence = np.array(prices[-5:])
            scaled_sequence = scaler.transform(last_sequence.reshape(-1, 1))
            scaled_sequence = scaled_sequence.reshape(1, 5, 1)
            
            scaled_prediction = model.predict(scaled_sequence)
            prediction = float(scaler.inverse_transform(scaled_prediction)[0][0])
            
            current_price = prices[-1]
            trend = "stable"
            if prediction > current_price * 1.05:
                trend = "increasing"
            elif prediction < current_price * 0.95:
                trend = "decreasing"
            
            return {
                'status': 'success',
                'product': product_name,
                'current_price': float(current_price),
                'predicted_price': round(prediction, 2),
                'trend': trend,
                'confidence_interval': [round(prediction * 0.95, 2), round(prediction * 1.05, 2)]
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

# Initialize predictor
price_predictor = PricePrediction('product_history.csv')