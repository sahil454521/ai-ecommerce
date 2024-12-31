from flask import Flask, request, jsonify


app = Flask(__name__)

@app.route('/create-barter', methods=['POST'])
def create_barter():
    barter_data = request.json  
    return jsonify({"message": "Barter offer created!"})

@app.route('/respond-barter', methods=['POST'])
def respond_barter():
    response_data = request.json 
    return jsonify({"message": f"Barter offer {response_data['status']}!"})