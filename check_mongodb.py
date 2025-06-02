from pymongo import MongoClient
from bson import ObjectId
import datetime

def check_mongodb():
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['cyberlaw_assistant']
    
    print("Connected to MongoDB")
    print("\nCollections in database:")
    print("-" * 50)
    
    # Get all collections
    collections = db.list_collection_names()
    
    for collection_name in collections:
        collection = db[collection_name]
        print(f"\nCollection: {collection_name}")
        print("-" * 30)
        
        # Get document count
        count = collection.count_documents({})
        print(f"Total documents: {count}")
        
        # Get sample documents (first 5)
        documents = list(collection.find().limit(5))
        
        if documents:
            print("\nSample documents:")
            for doc in documents:
                # Convert ObjectId to string and datetime to ISO format
                formatted_doc = {}
                for key, value in doc.items():
                    if isinstance(value, ObjectId):
                        formatted_doc[key] = str(value)
                    elif isinstance(value, datetime.datetime):
                        formatted_doc[key] = value.isoformat()
                    else:
                        formatted_doc[key] = value
                print(f"  {formatted_doc}")
        
        # Get collection indexes
        indexes = list(collection.list_indexes())
        if indexes:
            print("\nIndexes:")
            for index in indexes:
                print(f"  {index['name']}: {index['key']}")
        
        print("-" * 50)
    
    client.close()

if __name__ == "__main__":
    check_mongodb() 