"""
Verify data in MongoDB database
"""
from pymongo import MongoClient


def verify_database():
    """Check data in MongoDB"""
    print("Verifying MongoDB database...")
    
    client = MongoClient("mongodb://localhost:27017/")
    db = client["crypto_prediction"]
    
    print("\nCollections in database:")
    collections = db.list_collection_names()
    for collection in collections:
        count = db[collection].count_documents({})
        print(f"  - {collection}: {count} documents")
    
    # Check model_performance
    print("\nModel Performance Data:")
    performances = db.model_performance.find()
    for perf in performances:
        print(f"  Model: {perf['model_name']}")
        print(f"    Symbol: {perf['symbol']}")
        print(f"    Metric: {perf['metric_name']} = {perf['metric_value']}")
        print(f"    Training samples: {perf['training_samples']}")
    
    # Check predictions
    print("\nPredictions:")
    predictions = db.predictions.find()
    for pred in predictions:
        print(f"  Symbol: {pred['symbol']}")
        print(f"    Model: {pred['model_name']}")
        print(f"    Predicted price: {pred['predicted_price']}")
        print(f"    Target date: {pred['target_date']}")
        print(f"    Confidence: {pred['confidence']}")
    
    # Check system_logs
    print("\nSystem Logs:")
    logs = db.system_logs.find().sort("log_date", -1).limit(3)
    for log in logs:
        print(f"  [{log['log_level']}] {log['message']}")
        print(f"    Module: {log['module']}")
        print(f"    Details: {log['details']}")
    
    client.close()
    print("\nDatabase verification complete!")


if __name__ == "__main__":
    verify_database()
