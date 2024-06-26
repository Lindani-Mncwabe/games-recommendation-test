import pandas as pd
import numpy as np
from google.cloud import bigquery
from pandas_gbq import to_gbq
from recommendation import generate_recommendations
import logging
from time import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client = bigquery.Client()

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
        logging.info(f"Dropped table {table_id}")
    except Exception as e:
        logging.warning(f"Table {table_id} does not exist: {e}")

    table.clustering_fields = ["user_id"]
    table = client.create_table(table, exists_ok=True)
    logging.info(f"Created table {table.project}.{table.dataset_id}.{table.table_id}")

    df['GameID'] = df['GameID'].fillna(-1).astype(np.int64)
    df['City_Ranking'] = df['City_Ranking'].fillna(-1).astype(np.int64)
    df['user_id'] = df['user_id'].astype(str)
    df['country'] = df['country'].astype(str)
    df['city'] = df['city'].astype(str)
    df['recommendation_type'] = df['recommendation_type'].astype(str)
    df['recommendation_activity'] = df['recommendation_activity'].astype(str)

    try:
        logging.info("Starting to load data to BigQuery...")
        start_time = time()
        to_gbq(df, table_id, project_id=table.project, if_exists='append')
        end_time = time()
        logging.info(f'Data loading done in {end_time - start_time} seconds')
    except Exception as e:
        logging.error(f"Error loading data to BigQuery: {e}")
        raise

def manage_down_stream_update():
    merge_sql = """
        MERGE `ayoba-183a7.analytics_dev.test_rec_card_recommendations` AS A
            USING ( SELECT * FROM `ayoba-183a7.analytics_dev.test_rec_games_recommendations_activity_staging`
            WHERE GameID IS NOT NULL AND GameID != -1) AS B
        ON (
        A.user_id = B.user_id 
        AND A.GameID = B.GameID
        AND A.recommendation_type = B.recommendation_type
        AND A.recommendation_activity = B.recommendation_activity
        )
        WHEN MATCHED AND A.City_Ranking != B.City_Ranking THEN

        UPDATE SET 
            A.City_Ranking = B.City_Ranking

        WHEN NOT MATCHED BY TARGET THEN
        INSERT (user_id, country, city, GameID, City_Ranking, recommendation_type, recommendation_activity)
        VALUES (B.user_id, B.country, B.city, B.GameID, B.City_Ranking, B.recommendation_type, B.recommendation_activity)
        WHEN NOT MATCHED BY SOURCE THEN
        DELETE
    """
    try:
        logging.info("Starting downstream update...")
        start_time = time()
        query_job = client.query(merge_sql)
        query_job.result()
        end_time = time()
        logging.info(f"Downstream update completed in {end_time - start_time} seconds")
    except Exception as e:
        logging.error(f"Error in downstream update: {e}")
        raise

if __name__ == '__main__':
    try:
        logging.info("Starting recommendations process...")
        recommendations_df = generate_recommendations()
        logging.info("Recommendations DataFrame created")
        create_and_load_recommendations(recommendations_df)
        logging.info("Recommendations loaded to BigQuery")
        manage_down_stream_update()
        logging.info("Process completed successfully")
    except Exception as e:
        logging.error(f"Error in the recommendations process: {e}")
        raise