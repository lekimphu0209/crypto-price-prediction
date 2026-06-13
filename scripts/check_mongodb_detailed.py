"""
Detailed MongoDB Schema and Data Check
"""
from pymongo import MongoClient


def check_mongodb():
    """Check MongoDB schema and data"""
    print("=" * 80)
    print("MONGODB SCHEMA AND DATA ANALYSIS")
    print("=" * 80)
    
    client = MongoClient("mongodb://localhost:27017/")
    db = client["crypto_prediction"]
    
    collections = db.list_collection_names()
    print(f"\nCollections found: {len(collections)}")
    
    for collection_name in collections:
        collection = db[collection_name]
        count = collection.count_documents({})
        print(f"\n{'='*80}")
        print(f"Collection: {collection_name}")
        print(f"Total documents: {count}")
        print(f"{'='*80}")
        
        if count > 0:
            # Get sample document
            sample = collection.find_one()
            print(f"\nSample document structure:")
            for key, value in sample.items():
                print(f"  {key}: {type(value).__name__} = {value}")
            
            # Show all documents
            print(f"\nAll documents:")
            for doc in collection.find():
                print(f"  {doc}")
        else:
            print("  (empty)")
    
    client.close()
    print("\n" + "=" * 80)


if __name__ == "__main__":
    check_mongodb()
