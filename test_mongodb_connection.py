"""
Test MongoDB Connection
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from pymongo import MongoClient


def test_mongodb_connection():
    """Test MongoDB connection and create database"""
    try:
        # Connect to MongoDB
        print("🔌 Connecting to MongoDB...")
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        
        # Test connection
        client.admin.command('ping')
        print("✅ MongoDB connection successful!")
        
        # Create/use database
        db = client["crypto_prediction"]
        print(f"✅ Database 'crypto_prediction' ready")
        
        # Create test collection and insert data
        test_collection = db.test_collection
        test_collection.insert_one({"test": "connection", "timestamp": "now"})
        print("✅ Test data inserted successfully")
        
        # Query test data
        result = test_collection.find_one({"test": "connection"})
        print(f"✅ Test data retrieved: {result}")
        
        # Clean up
        test_collection.delete_one({"test": "connection"})
        print("✅ Test data cleaned up")
        
        # List all collections
        collections = db.list_collection_names()
        print(f"📋 Collections in database: {collections}")
        
        client.close()
        print("\n🎉 MongoDB setup complete!")
        return True
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check if MongoDB service is running:")
        print("   - Windows: Get-Service MongoDB")
        print("   - Start service: Start-Service MongoDB")
        print("2. Check if MongoDB is installed")
        print("3. Check if port 27017 is available")
        return False


if __name__ == "__main__":
    test_mongodb_connection()
