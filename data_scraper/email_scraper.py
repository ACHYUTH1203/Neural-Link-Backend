import pandas as pd
from pymongo import MongoClient

def upload_to_mongo():
    # 1. Configuration
    # If you are using MongoDB Atlas, replace the string below with your Atlas URI
    mongo_uri = "mongodb://localhost:27017/"
    db_name = "Elon"
    
    # Define your file paths
    posts_path = r"C:\Users\achyu\Desktop\all_musk_posts.csv"
    threads_path = r"C:\Users\achyu\Desktop\musk_quote_tweets.csv"

    try:
        # 2. Connect to MongoDB
        client = MongoClient(mongo_uri)
        db = client[db_name]
        print("Connected to MongoDB successfully.")

        # 3. Process 'all_musk_posts.csv' -> 'posts' collection
        print("Reading all_musk_posts.csv...")
        df_posts = pd.read_csv(posts_path)
        # Convert NaN values to None (MongoDB handles 'null' better than 'NaN')
        posts_data = df_posts.where(pd.notnull(df_posts), None).to_dict(orient='records')
        
        db.posts.insert_many(posts_data)
        print(f"Successfully uploaded {len(posts_data)} records to 'posts' collection.")

        # 4. Process 'musk_quote_tweets.csv' -> 'threads' collection
        print("Reading musk_quote_tweets.csv...")
        df_threads = pd.read_csv(threads_path)
        threads_data = df_threads.where(pd.notnull(df_threads), None).to_dict(orient='records')
        
        db.threads.insert_many(threads_data)
        print(f"Successfully uploaded {len(threads_data)} records to 'threads' collection.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    upload_to_mongo()