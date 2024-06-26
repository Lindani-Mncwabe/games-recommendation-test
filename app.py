from flask import Flask, jsonify, request
from recommendation import generate_recommendations, get_last_played_game

app = Flask(__name__)

try:
    recommendations_df = generate_recommendations()
except Exception as e:
    app.logger.error(f"Error generating recommendations: {e}")
    recommendations_df = None

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        if recommendations_df is None:
            return jsonify({"error": "Recommendations are not available at the moment"}), 500

        data = request.get_json()
        user_id = str(data.get('user_id', '')).strip()

        if not user_id:
            return jsonify({"error": "User ID must be provided"}), 400

        last_played_game = get_last_played_game(user_id)
        if not last_played_game:
            return jsonify({"error": f"No last played game found for user '{user_id}'"}), 404

        user_recommendations = recommendations_df[recommendations_df['user_id'] == user_id]
        if user_recommendations.empty:
            return jsonify({"error": f"No recommendations found for user '{user_id}'"}), 404

        recommendations = user_recommendations.to_dict(orient='records')
        return jsonify({
            "last_played_game": last_played_game,
            "recommendations": recommendations
        })
    except Exception as e:
        app.logger.error(f"Error in recommendation: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
