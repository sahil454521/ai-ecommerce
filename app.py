from flask import Flask, jsonify, request
# from flask cors import CORS  # Recommended for handling cross-origin requests
from leaderboard import leaderboard
from batering import create_barter, respond_barter
from mainai import predict_price

app = Flask(__name__)
# CORS(app)  # This helps prevent CORS issues

# Your existing routes remain the same

@app.route('/leaderboard', methods=['GET'])
def leaderboard_route():
    return leaderboard()


@app.route('/create-barter', methods=['POST'])
def create_barter_route():
    return create_barter()

@app.route('/respond-barter', methods=['POST'])
def respond_barter_route():
    return respond_barter()


@app.route('/predict-price', methods=['POST'])
def predict_price_route():
    return predict_price()

if __name__ == '__main__':
    app.run(debug=True)
