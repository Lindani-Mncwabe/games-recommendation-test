import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from data_processing import df_all_available_games, df_user_last_game_played
import logging

# Initialize TF-IDF Vectorizer and compute cosine similarity matrix
try:
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_all_available_games['game_title_processed'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
except Exception as e:
    logging.error(f"Error initializing TF-IDF vectorizer or computing cosine similarity: {e}")
    raise

def get_content_based_recommendations(game_name, df_all_available_games=df_all_available_games, cosine_sim=cosine_sim):
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
        logging.error(f"Error getting content-based recommendations for game '{game_name}': {e}")
        raise

def get_recommendations_df():
    try:
        distinct_games = df_user_last_game_played['game_processed'].drop_duplicates().dropna()
        recommendations_list = []

        for game_name in distinct_games:
            recommendations = get_content_based_recommendations(game_name)
            for rank, recommendation in enumerate(recommendations, start=1):
                recommendations_list.append({
                    'game_name': game_name,
                    'recommendation_game': recommendation,
                    'ranking': rank
                })

        recommendations_df = pd.DataFrame(recommendations_list)
        return recommendations_df
    except Exception as e:
        logging.error(f"Error generating recommendations DataFrame: {e}")
        raise

def pre_process_recommendation(recommendations_df):
    try:
        df_reco = recommendations_df
        df_merged_reco = pd.merge(df_reco, df_all_available_games, left_on='recommendation_game', right_on='game_title_processed', how='left')
        df_user_rec = df_user_last_game_played[['user_id', 'game_processed']]
        df_user_games_recommendations = pd.merge(df_user_rec, df_merged_reco, left_on='game_processed', right_on='game_name', how='left')

        df_user_games_recommendations['ranking'].fillna(-1, inplace=True)
        df_user_games_recommendations['ranking'] = df_user_games_recommendations['ranking'].astype(int)
        df_user_games_recommendations['country'] = ''
        df_user_games_recommendations['city'] = ''
        df_user_games_recommendations['recommendation_type'] = 'games'
        df_user_games_recommendations['recommendation_activity'] = 'user_activity'
        df_user_games_recommendations.rename(columns={'game_id': 'GameID'}, inplace=True)
        df_user_games_recommendations.rename(columns={'ranking': 'City_Ranking'}, inplace=True)

        return df_user_games_recommendations[['user_id', 'country', 'city', 'GameID', 'City_Ranking', 'recommendation_type', 'recommendation_activity']]
    except Exception as e:
        logging.error(f"Error in preprocessing recommendations: {e}")
        raise

def generate_recommendations():
    try:
        logging.info("Generating content-based recommendations...")
        recommendations_df = get_recommendations_df()
        logging.info("Content-based recommendations generated.")
        return pre_process_recommendation(recommendations_df)
    except Exception as e:
        logging.error(f"Error generating recommendations: {e}")
        raise

def get_last_played_game(user_id, df_user_last_game_played=df_user_last_game_played):
    try:
        user_game = df_user_last_game_played[df_user_last_game_played['user_id'] == user_id]['game'].values
        if len(user_game) == 0:
            return None
        return user_game[0]
    except Exception as e:
        logging.error(f"Error getting last played game for user '{user_id}': {e}")
        raise