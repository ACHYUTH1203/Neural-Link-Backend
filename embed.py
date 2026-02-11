import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

client = MongoClient("mongodb://localhost:27017/")
db = client["Elon"]

model = SentenceTransformer('all-MiniLM-L6-v2')
collections_to_process = ["books", "frameworks", "podcasts", "threads"]

def migrate_to_vectors():
    for coll_name in collections_to_process:
        collection = db[coll_name]
        print(f" Processing collection: {coll_name}")
        query = {"embedding": {"$exists": False}}
        documents = list(collection.find(query))
        
        if not documents:
            print(f" All documents in {coll_name} already have embeddings.")
            continue

        for doc in tqdm(documents, desc=f"Embedding {coll_name}"):

            text_content = doc.get("content") or doc.get("orig_tweet_text") or doc.get("title")
            
            if text_content:
                if isinstance(text_content, list):
                    text_content = " ".join([str(i) for i in text_content])
                vector = model.encode(text_content).tolist()
                collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"embedding": vector}}
                )

if __name__ == "__main__":
    migrate_to_vectors()
    print(" Phase 1 Complete: All collections are now vectorized.")