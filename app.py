import os
import re
import string
import unicodedata

import contractions
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request
from google.cloud import bigquery
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pandas_gbq import to_gbq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from textblob import TextBlob
from unidecode import unidecode

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'ayb_gcs_credentials.json'

# Initialize the BigQuery client
client = bigquery.Client()

# Define the SQL query
sql_query_last_game_played = """
WITH ranked_entries AS (
  SELECT *,
         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_date DESC) AS rank
  FROM `ayoba-183a7.analytics_dw.user_daily_games`
  WHERE event_date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE()
)
SELECT *
FROM ranked_entries
WHERE rank = 1 
  AND game IS NOT NULL 
  AND game != '';
"""

# Define the SQL query
sql_query_all_available_games = """
SELECT distinct game_title,game_id FROM `ayoba-183a7.analytics_dw.dim_games` 
WHERE game_title IS NOT NULL AND game_title != '';
"""

def get_bigquery_data(client, sql_query):
    # Run the query
    query_job = client.query(sql_query)

    # Get the results as a DataFrame
    df = query_job.to_dataframe()

    # Display the DataFrame
    return df
    
    
df_user_last_game_played = get_bigquery_data(client, sql_query_last_game_played)
df_all_available_games = get_bigquery_data(client, sql_query_all_available_games)

class NltkPreprocessingSteps:
    def __init__(self, X):
        self.X = X
        self.sw_nltk = stopwords.words('english')
        new_stopwords = ['<*>','Ayoba','ayoba']
        self.sw_nltk.extend(new_stopwords)
        self.remove_punctuations = string.punctuation.replace('.','')

    def remove_html_tags(self):
        try:
            self.X = self.X.apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
            return self
        except Exception as e:
            app.logger.error(f"Error removing HTML tags: {e}")
            raise

    def remove_accented_chars(self):
        try:
            self.X = self.X.apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore'))
            return self
        except Exception as e:
            app.logger.error(f"Error removing accented characters: {e}")
            raise

    def replace_diacritics(self):
        try:
            self.X = self.X.apply(lambda x: unidecode(x, errors="preserve"))
            return self
        except Exception as e:
            app.logger.error(f"Error replacing diacritics: {e}")
            raise

    def to_lower(self):
        try:
            self.X = self.X.apply(lambda x: " ".join([word.lower() for word in x.split() if word and word not in self.sw_nltk]) if x else '')
            return self
        except Exception as e:
            app.logger.error(f"Error converting to lower case: {e}")
            raise

    def expand_contractions(self):
        try:
            self.X = self.X.apply(lambda x: " ".join([contractions.fix(expanded_word) for expanded_word in x.split()]))
            return self
        except Exception as e:
            app.logger.error(f"Error expanding contractions: {e}")
            raise

    def remove_numbers(self):
        try:
            self.X = self.X.apply(lambda x: re.sub(r'\d+', '', x))
            return self
        except Exception as e:
            app.logger.error(f"Error removing numbers: {e}")
            raise

    def remove_http(self):
        try:
            self.X = self.X.apply(lambda x: re.sub(r'http\S+', '', x))
            return self
        except Exception as e:
            app.logger.error(f"Error removing http links: {e}")
            raise
    
    def remove_words_with_numbers(self):
        try:
            self.X = self.X.apply(lambda x: re.sub(r'\w*\d\w*', '', x))
            return self
        except Exception as e:
            app.logger.error(f"Error removing words with numbers: {e}")
            raise
    
    def remove_digits(self):
        try:
            self.X = self.X.apply(lambda x: re.sub(r'[0-9]+', '', x))
            return self
        except Exception as e:
            app.logger.error(f"Error removing digits: {e}")
            raise
    
    def remove_special_character(self):
        try:
            self.X = self.X.apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]+', ' ', x))
            return self
        except Exception as e:
            app.logger.error(f"Error removing special characters: {e}")
            raise
    
    def remove_white_spaces(self):
        try:
            self.X = self.X.apply(lambda x: re.sub(r'\s+', ' ', x).strip())
            return self
        except Exception as e:
            app.logger.error(f"Error removing white spaces: {e}")
            raise
    
    def remove_extra_newlines(self):
        try:
            self.X == self.X.apply(lambda x: re.sub(r'[\r|\n|\r\n]+', ' ', x))
            return self
        except Exception as e:
            app.logger.error(f"Error removing extra newlines: {e}")
            raise

    def replace_dots_with_spaces(self):
        try:
            self.X = self.X.apply(lambda x: re.sub("[.]", " ", x))
            return self
        except Exception as e:
            app.logger.error(f"Error replacing dots with spaces: {e}")
            raise

    def remove_punctuations_except_periods(self):
        try:
            self.X = self.X.apply(lambda x: re.sub('[%s]' % re.escape(self.remove_punctuations), '' , x))
            return self
        except Exception as e:
            app.logger.error(f"Error removing punctuations except periods: {e}")
            raise

    def remove_all_punctuations(self):
        try:
            self.X = self.X.apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))
            return self
        except Exception as e:
            app.logger.error(f"Error removing all punctuations: {e}")
            raise

    def remove_double_spaces(self):
        try:
            self.X = self.X.apply(lambda x: re.sub(' +', '  ', x))
            return self
        except Exception as e:
            app.logger.error(f"Error removing double spaces: {e}")
            raise

    def fix_typos(self):
        try:
            self.X = self.X.apply(lambda x: str(TextBlob(x).correct()))
            return self
        except Exception as e:
            app.logger.error(f"Error fixing typos: {e}")
            raise

    def remove_stopwords(self):
        try:
            self.X = self.X.apply(lambda x: " ".join([ word for word in x.split() if word not in self.sw_nltk]))
            return self
        except Exception as e:
            app.logger.error(f"Error removing stopwords: {e}")
            raise
    
    def remove_singleChar(self):
        try:
            self.X = self.X.apply(lambda x: " ".join([ word for word in x.split() if len(word)>2]))
            return self
        except Exception as e:
            app.logger.error(f"Error removing single characters: {e}")
            raise

    def lemmatize(self):
        try:
            lemmatizer = WordNetLemmatizer()
            self.X = self.X.apply(lambda x: " ".join([ lemmatizer.lemmatize(word) for word in x.split()]))
            return self
        except Exception as e:
            app.logger.error(f"Error lemmatizing: {e}")
            raise

    def get_processed_text(self):
        return self.X

# Preprocess data
try:
    txt_preproc_all_games = NltkPreprocessingSteps(df_all_available_games['game_title'])
    txt_preproc_user_games = NltkPreprocessingSteps(df_user_last_game_played['game'])

    processed_text_all_games = txt_preproc_all_games.to_lower().remove_html_tags().remove_accented_chars().replace_diacritics().expand_contractions().remove_numbers().remove_digits().remove_special_character().remove_white_spaces().remove_extra_newlines().replace_dots_with_spaces().remove_punctuations_except_periods().remove_words_with_numbers().remove_singleChar().remove_double_spaces().lemmatize().remove_stopwords().get_processed_text()
    processed_text_all_users = txt_preproc_user_games.to_lower().remove_html_tags().remove_accented_chars().replace_diacritics().expand_contractions().remove_numbers().remove_digits().remove_special_character().remove_white_spaces().remove_extra_newlines().replace_dots_with_spaces().remove_punctuations_except_periods().remove_words_with_numbers().remove_singleChar().remove_double_spaces().lemmatize().remove_stopwords().get_processed_text()

    df_all_available_games['game_title_processed'] = processed_text_all_games
    df_user_last_game_played['game_processed'] = processed_text_all_users
except Exception as e:
    app.logger.error(f"Error in data preprocessing: {e}")
    raise

# TF-IDF vectorization
try:
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_all_available_games['game_title_processed'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
except Exception as e:
    app.logger.error(f"Error in TF-IDF vectorization: {e}")
    raise

def get_content_based_recommendations(game_name, cosine_sim=cosine_sim, df_all_available_games=df_all_available_games):
    try:
        idx = df_all_available_games[df_all_available_games['game_title_processed'] == game_name].index
        if len(idx) == 0:
            return []
        else:
            idx = idx[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]  # Top 10 similar games
            similar_games = [df_all_available_games['game_title_processed'][i[0]] for i in sim_scores]
            return similar_games
    except Exception as e:
        app.logger.error(f"Error in getting content-based recommendations: {e}")
        raise

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        user_id = str(data.get('user_id', '')).strip()

        if not user_id:
            return jsonify({"error": "User ID must be provided"}), 400

        # Retrieve last played game for the user
        user_game = df_user_last_game_played[df_user_last_game_played['user_id'] == user_id]['game_processed'].values
        if len(user_game) == 0:
            return jsonify({"error": f"No last played game found for user '{user_id}'"}), 404

        game_name = user_game[0]

        # Get recommendations
        idx = df_all_available_games[df_all_available_games['game_title_processed'] == game_name].index
        if len(idx) == 0:
            return jsonify({"error": f"No similar games found for the last played game '{game_name}'"}), 404

        idx = idx[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]  # Top 10

        recommendations = []
        for rank, i in enumerate(sim_scores, start=1):
            game_info = df_all_available_games.iloc[i[0]]
            recommendations.append({
                "user_id": str(user_id),
                "GameID": int(game_info['game_id']),
                "country": game_info.get('country', ''),
                "city": game_info.get('city', ''),
                "City_Ranking": rank,  # Rank from 1 to 10
                "recommendation_type": "games",
                "recommendation_activity": "user_activity"
            })

        # Load recommendations to BQ
        # recommendations_df = pd.DataFrame(recommendations)
        # create_and_load_recommendations(recommendations_df)
        # manage_down_stream_update()

        return jsonify({
            "last_played_game": game_name,
            "recommendations": recommendations
        })
    except Exception as e:
        app.logger.error(f"Error in recommendation: {e}")
        return jsonify({"error": "Internal server error"}), 500

def create_and_load_recommendations(df):
    schema = [
        bigquery.SchemaField("user_id", "STRING"),
        bigquery.SchemaField("country", "STRING"),
        bigquery.SchemaField("city", "STRING"),
        bigquery.SchemaField("GameID", "INTEGER"),
        bigquery.SchemaField("City_Ranking", "INTEGER"),
        bigquery.SchemaField("recommendation_type", "STRING"),
        bigquery.SchemaField("recommendation_activity", "STRING"),
    ]

    dataset_id = 'ayoba-183a7.analytics_dev' 
    table_id = f'{dataset_id}.test_rec_games_recommendations_activity_staging'
    table = bigquery.Table(table_id, schema=schema)

    try:
        client.delete_table(table)
        print(f"Dropped table {table_id}")
    except Exception as e:
        app.logger.warning(f"Table {table_id} does not exist: {e}")

    table.clustering_fields = ["user_id"]
    table = client.create_table(table, exists_ok=True)
    print(f"Created table {table.project}.{table.dataset_id}.{table.table_id}")

    df['GameID'] = df['GameID'].fillna(-1).astype(np.int64)
    df['City_Ranking'] = df['City_Ranking'].fillna(-1).astype(np.int64)
    df['user_id'] = df['user_id'].astype(str)
    df['country'] = df['country'].astype(str)
    df['city'] = df['city'].astype(str)
    df['recommendation_type'] = df['recommendation_type'].astype(str)
    df['recommendation_activity'] = df['recommendation_activity'].astype(str)

    try:
        to_gbq(df, table_id, project_id=table.project, if_exists='append')
        print('Data loading done')
    except Exception as e:
        app.logger.error(f"Error loading data to BigQuery: {e}")
        raise

def manage_down_stream_update():
    merge_sql = """
    MERGE `ayoba-183a7.analytics_dw.test_rec_card_recommendations` AS A
    USING (SELECT * FROM `ayoba-183a7.analytics_dev.test_rec_games_recommendations_activity_staging` 
            where GameId is not null or GameId !=-1)  AS B
    ON (A.user_id = B.user_id 
        AND A.GameID = B.GameID
        AND B.recommendation_type = B.recommendation_type
        AND B.recommendation_activity)
    WHEN MATCHED AND A.ranking != B.ranking THEN
      UPDATE SET 

        A.ranking = B.ranking,

    WHEN NOT MATCHED BY TARGET THEN
      INSERT (user_id, country, city, GameID, City_Ranking, recommendation_type, recommendation_activity)
      VALUES (B.user_id, B.country, B.city, B.GameID, B.City_Ranking, B.recommendation_type, B.recommendation_activity)
    WHEN NOT MATCHED BY SOURCE 
        AND recommendation_type = 'user_activity' 
        AND recommendation_type = 'games' THEN
      DELETE
    """
    try:
        query_job = client.query(merge_sql)
        query_job.result()
    except Exception as e:
        app.logger.error(f"Error in downstream update: {e}")
        raise

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
