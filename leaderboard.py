from flask import jsonify

def leaderboard():
  
    leaderboard = [
        {"seller_id": 1, "name": "Shopkeeper A", "total_sales": 10000},
        {"seller_id": 2, "name": "Shopkeeper B", "total_sales": 9000}
    ]
    return jsonify({"leaderboard": leaderboard})