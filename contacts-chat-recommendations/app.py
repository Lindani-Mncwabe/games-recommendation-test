from flask import Flask, render_template, request, url_for, jsonify, json, Blueprint, send_file, make_response
import numpy as np
import pandas as pd
import logging
from flasgger import Swagger
import os
from google.cloud import bigquery
import io
import base64
import matplotlib.pyplot as plt
import networkx as nx
from adjustText import adjust_text

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for rendering
import matplotlib.pyplot as plt

# Initialize Flask app with Swagger
app = Flask(__name__)
swagger = Swagger(app)  

# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()  
stream_handler.setFormatter(log_formatter)  
app.logger.addHandler(stream_handler)  
app.logger.setLevel(logging.INFO)

# Ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not credentials_path:
    raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set!")
if not os.path.isfile(credentials_path):
    raise FileNotFoundError(f"Credentials file not found at {credentials_path}")

# Initialize BigQuery client
bigquery_client = bigquery.Client()

# Define Flask routes
@app.route('/')
def index():
    """
    Home route
    ---
    responses:
      200:
        description: Welcome message
    """
    return 'Hello, recommendation!'


@app.route('/health')
def health():
    logging.info('Health route accessed')
    return jsonify({"status": "ok"})

@app.route('/get-data', methods=['GET'])
def get_data():
    logging.info('Get-data route accessed')
    data = {"key": "value"}
    return jsonify(data)

@app.route('/post-data', methods=['POST'])
def post_data():
    logging.info('POST /post-data route accessed')
    posted_data = request.get_json()
    response_data = {"received": posted_data}
    return jsonify(response_data)

def visualize_recommendations(df_ranked_recommendations, buffer):
    try:
        app.logger.info("Visualizing recommendations as a social graph.")
        G = nx.DiGraph()
        for _, row in df_ranked_recommendations.iterrows():
            G.add_edge(row['user_id'], row['recommended_contact_user_id'], weight=row['ranking'])
        
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=0.5, seed=42)
        receiving_users = set(df_ranked_recommendations['user_id'].unique())
        node_colors = ['red' if node in receiving_users else 'lightblue' for node in G.nodes()]
        
        nx.draw(G, pos, with_labels=False, node_size=800, node_color=node_colors, font_size=10)
        texts = [plt.text(x, y, node, fontsize=9, weight='bold', ha='center', va='center') for node, (x, y) in pos.items()]
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.savefig(buffer, format='png')
        plt.close()
        
    except Exception as e:
        app.logger.error(f"Error visualizing social graph: {e}")

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    """
    Fetch Chat Recommendations for One or More Users
    ---
    tags:
      - Chat Recommendations
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - user_ids
          properties:
            user_ids:
              type: array
              items:
                type: string
              description: A list of user IDs to fetch chat recommendations for. Each ID should be a valid phone number starting with '+'.
              example: ["+2250545356890", "+1234567890"]
    responses:
      200:
        description: List of chat recommendations and visualization.
        content:
          application/json:
            schema:
              type: object
              properties:
                data:
                  type: array
                  items:
                    type: object
                    properties:
                      userJid:
                        type: string
                        description: The user's JID (Jabber ID).
                      user_id:
                        type: string
                        description: The ID of the user.
                      recommended_contact_user_id:
                        type: string
                        description: The recommended contact's user ID.
                      mutual_contacts:
                        type: integer
                        description: Number of mutual contacts.
                      ranking:
                        type: integer
                        description: The ranking of the recommendation.
          image/png:
            schema:
              type: string
              format: binary
              description: Social graph visualization.
    """
    try:
        app.logger.info('Predict data endpoint called')
        
        data = json.loads(request.data)
        user_ids = data.get('user_ids')
        
        if not user_ids or not isinstance(user_ids, list):
            return jsonify({"error": "user_ids must be provided as a list"}), 400
        
        user_ids = [f"+{uid}" if "+" not in uid else uid for uid in user_ids]
        
        query = """
            SELECT userJid, user_id, recommended_contact_user_id, mutual_contacts, ranking
            FROM `ayoba-183a7.analytics_dw.contact_chat_recommendations`
            WHERE user_id IN UNNEST(@user_ids)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("user_ids", "STRING", user_ids)]
        )
        
        query_job = bigquery_client.query(query, job_config=job_config)
        records = [dict(row) for row in query_job]

        if not records:
            return jsonify([]), 200

        df_ranked_recommendations = pd.DataFrame(records)

        # Visualize the recommendations
        buffer = io.BytesIO()
        visualize_recommendations(df_ranked_recommendations, buffer)
        
        buffer.seek(0)
        response = make_response(send_file(buffer, mimetype='image/png'))
        response.headers["Content-Disposition"] = "inline; filename=graph.png"
        return response

    except Exception as e:
        app.logger.error(f"Error in predict_datapoint: {e}")
        return jsonify({"error": "Internal server error"}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
