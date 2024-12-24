from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)
# Initialize Flask app
app = Flask(__name__)

# Initialize models and scalers for individual products
models = {}
scalers = {}
product_history = {
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

def initialize_models():
    for product_name, details in product_history.items():
        create_model(product_name)

def create_model(product_name):
    prices = product_history[product_name]['prices']

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
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)

    models[product_name] = model
    scalers[product_name] = scaler

initialize_models()

@app.route('/predict-price', methods=['POST'])
def predict_price():
    try:
        data = request.json
        
        product_name = data.get('product_name')
        print("Received product name:", data.get('product_name'))

        if product_name not in product_history:
            return jsonify({"error": "Product not found", "status": "error"}), 404

        prices = product_history[product_name]['prices']
        scaler = scalers[product_name]
        model = models[product_name]

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

        # Include additional metrics
        response = {
            'status': 'success',
            'product': product_name,
            'current_price': current_price,
            'predicted_price': round(prediction, 2),
            'trend': trend,
            'history': prices[-5:],
            'product_id': product_history[product_name]['id'],
            'confidence_interval': [round(prediction * 0.95, 2), round(prediction * 1.05, 2)]
        }

        return jsonify(response)

    except Exception as e:
        print("Error in /predict-price:", str(e))
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route('/leaderboard', methods=['GET'])
def leaderboard():
    leaderboard_data = [
        {"seller": "Shop A", "total_sales": 15000},
        {"seller": "Shop B", "total_sales": 12000},
        {"seller": "Shop C", "total_sales": 10000}
    ]
    return jsonify({"leaderboard": leaderboard_data})

@app.route('/create-barter', methods=['POST'])
def create_barter():
    try:
        barter_data = request.json
        return jsonify({"message": "Barter offer created successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
