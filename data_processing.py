import os
import re
import string
import unicodedata

import contractions
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from google.cloud import bigquery
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/Users/lindani/Documents/Service-Accounts/ayb_gcs_credentials.json'

# Initialize the BigQuery client
client = bigquery.Client()

def get_bigquery_data(client, sql_query):
    try:
        query_job = client.query(sql_query)
        df = query_job.to_dataframe()
        return df
    except Exception as e:
        logging.error(f"Error fetching data from BigQuery: {e}")
        raise

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

sql_query_all_available_games = """
SELECT distinct game_title,game_id FROM `ayoba-183a7.analytics_dw.dim_games` 
WHERE game_title IS NOT NULL AND game_title != '';
"""

try:
    df_user_last_game_played = get_bigquery_data(client, sql_query_last_game_played)
    df_all_available_games = get_bigquery_data(client, sql_query_all_available_games)
except Exception as e:
    logging.error(f"Error initializing dataframes: {e}")
    raise

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
            logging.error(f"Error removing HTML tags: {e}")
            raise

    def remove_accented_chars(self):
        try:
            self.X = self.X.apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore'))
            return self
        except Exception as e:
            logging.error(f"Error removing accented characters: {e}")
            raise

    def replace_diacritics(self):
        try:
            self.X = self.X.apply(lambda x: unidecode(x, errors="preserve"))
            return self
        except Exception as e:
            logging.error(f"Error replacing diacritics: {e}")
            raise

    def to_lower(self):
        try:
            self.X = self.X.apply(lambda x: " ".join([word.lower() for word in x.split() if word and word not in self.sw_nltk]) if x else '')
            return self
        except Exception as e:
            logging.error(f"Error converting to lower case: {e}")
            raise

    def expand_contractions(self):
        try:
            self.X = self.X.apply(lambda x: " ".join([contractions.fix(expanded_word) for expanded_word in x.split()]))
            return self
        except Exception as e:
            logging.error(f"Error expanding contractions: {e}")
            raise

    def remove_numbers(self):
        try:
            self.X = self.X.apply(lambda x: re.sub(r'\d+', '', x))
            return self
        except Exception as e:
            logging.error(f"Error removing numbers: {e}")
            raise

    def remove_http(self):
        try:
            self.X = self.X.apply(lambda x: re.sub(r'http\S+', '', x))
            return self
        except Exception as e:
            logging.error(f"Error removing http links: {e}")
            raise
    
    def remove_words_with_numbers(self):
        try:
            self.X = self.X.apply(lambda x: re.sub(r'\w*\d\w*', '', x))
            return self
        except Exception as e:
            logging.error(f"Error removing words with numbers: {e}")
            raise
    
    def remove_digits(self):
        try:
            self.X = self.X.apply(lambda x: re.sub(r'[0-9]+', '', x))
            return self
        except Exception as e:
            logging.error(f"Error removing digits: {e}")
            raise
    
    def remove_special_character(self):
        try:
            self.X = self.X.apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]+', ' ', x))
            return self
        except Exception as e:
            logging.error(f"Error removing special characters: {e}")
            raise
    
    def remove_white_spaces(self):
        try:
            self.X = self.X.apply(lambda x: re.sub(r'\s+', ' ', x).strip())
            return self
        except Exception as e:
            logging.error(f"Error removing white spaces: {e}")
            raise
    
    def remove_extra_newlines(self):
        try:
            self.X = self.X.apply(lambda x: re.sub(r'[\r|\n|\r\n]+', ' ', x))
            return self
        except Exception as e:
            logging.error(f"Error removing extra newlines: {e}")
            raise

    def replace_dots_with_spaces(self):
        try:
            self.X = self.X.apply(lambda x: re.sub("[.]", " ", x))
            return self
        except Exception as e:
            logging.error(f"Error replacing dots with spaces: {e}")
            raise

    def remove_punctuations_except_periods(self):
        try:
            self.X = self.X.apply(lambda x: re.sub('[%s]' % re.escape(self.remove_punctuations), '' , x))
            return self
        except Exception as e:
            logging.error(f"Error removing punctuations except periods: {e}")
            raise

    def remove_all_punctuations(self):
        try:
            self.X = self.X.apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))
            return self
        except Exception as e:
            logging.error(f"Error removing all punctuations: {e}")
            raise

    def remove_double_spaces(self):
        try:
            self.X = self.X.apply(lambda x: re.sub(' +', '  ', x))
            return self
        except Exception as e:
            logging.error(f"Error removing double spaces: {e}")
            raise

    def fix_typos(self):
        try:
            self.X = self.X.apply(lambda x: str(TextBlob(x).correct()))
            return self
        except Exception as e:
            logging.error(f"Error fixing typos: {e}")
            raise

    def remove_stopwords(self):
        try:
            self.X = self.X.apply(lambda x: " ".join([ word for word in x.split() if word not in self.sw_nltk]))
            return self
        except Exception as e:
            logging.error(f"Error removing stopwords: {e}")
            raise
    
    def remove_singleChar(self):
        try:
            self.X = self.X.apply(lambda x: " ".join([ word for word in x.split() if len(word)>2]))
            return self
        except Exception as e:
            logging.error(f"Error removing single characters: {e}")
            raise

    def lemmatize(self):
        try:
            lemmatizer = WordNetLemmatizer()
            self.X = self.X.apply(lambda x: " ".join([ lemmatizer.lemmatize(word) for word in x.split()]))
            return self
        except Exception as e:
            logging.error(f"Error lemmatizing: {e}")
            raise

    def get_processed_text(self):
        return self.X

try:
    txt_preproc_all_games = NltkPreprocessingSteps(df_all_available_games['game_title'])
    txt_preproc_user_games = NltkPreprocessingSteps(df_user_last_game_played['game'])

    processed_text_all_games = txt_preproc_all_games.to_lower().remove_html_tags().remove_accented_chars().replace_diacritics().expand_contractions().remove_numbers().remove_digits().remove_special_character().remove_white_spaces().remove_extra_newlines().replace_dots_with_spaces().remove_punctuations_except_periods().remove_words_with_numbers().remove_singleChar().remove_double_spaces().lemmatize().remove_stopwords().get_processed_text()
    processed_text_all_users = txt_preproc_user_games.to_lower().remove_html_tags().remove_accented_chars().replace_diacritics().expand_contractions().remove_numbers().remove_digits().remove_special_character().remove_white_spaces().remove_extra_newlines().replace_dots_with_spaces().remove_punctuations_except_periods().remove_words_with_numbers().remove_singleChar().remove_double_spaces().lemmatize().remove_stopwords().get_processed_text()

    df_all_available_games['game_title_processed'] = processed_text_all_games
    df_user_last_game_played['game_processed'] = processed_text_all_users
except Exception as e:
    logging.error(f"Error in preprocessing: {e}")
    raise