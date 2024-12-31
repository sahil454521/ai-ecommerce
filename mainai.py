from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pridiction import PricePrediction
import logging


logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Enable CORS

# Initialize the predictor
price_predictor = PricePrediction('product_history.csv')

@app.route('/', methods=['GET'])
def serve_index():
    return send_from_directory('.', 'index3.html')

@app.route('/<path:path>', methods=['GET'])
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/predict-price', methods=['POST'])
@app.route('/predict-price', methods=['POST'])
def predict_price():
    try:
        data = request.json
        print(f"Received data: {data}")  # Basic console logging
        
        if not data or 'product_name' not in data:
            return jsonify({'status': 'error', 'error': 'No product name provided'}), 400
            
        product_name = data['product_name']
        print(f"Processing product: {product_name}")
        print(f"Available products: {price_predictor.product_history.keys()}")
        
        prediction_result = price_predictor.predict_price(product_name)
        print(f"Prediction result: {prediction_result}")
        
        return jsonify(prediction_result)
        
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'status': 'error', 'error': str(e)}), 500
        


@app.route('/leaderboard', methods=['GET', 'OPTIONS'])
def get_leaderboard():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
        
    try:
        # Example leaderboard data
        leaderboard_data = [
            {"seller": "Shop A", "total_sales": 15000},
            {"seller": "Shop B", "total_sales": 12000},
            {"seller": "Shop C", "total_sales": 10000}
        ]
        return jsonify({"leaderboard": leaderboard_data, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
