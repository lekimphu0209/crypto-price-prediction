"""
Database Factory for Dependency Injection
Allows switching between SQL and MongoDB via configuration
"""
import os
from typing import Optional

from src.infrastructure.repositories.sql_repository import SQLDatabaseRepository
from src.infrastructure.repositories.mongo_repository import MongoDatabaseRepository


class DatabaseFactory:
    """Factory for creating database repositories"""
    
    @staticmethod
    def create_repository(db_type: str = None, 
                         connection_string: str = None) -> object:
        """
        Create database repository based on configuration
        
        Args:
            db_type: 'sql' or 'mongo' (default from env var or 'sql')
            connection_string: Database connection string
            
        Returns:
            Database repository instance
        """
        # Get database type from environment variable or parameter
        db_type = db_type or os.getenv('DB_TYPE', 'sql')
        
        if db_type.lower() == 'mongo':
            # MongoDB configuration
            mongo_connection = connection_string or os.getenv(
                'MONGO_CONNECTION_STRING', 
                'mongodb://localhost:27017/'
            )
            mongo_db = os.getenv('MONGO_DATABASE', 'crypto_prediction')
            
            return MongoDatabaseRepository(
                connection_string=mongo_connection,
                database_name=mongo_db
            )
        
        else:
            # SQL (default)
            return SQLDatabaseRepository()
    
    @staticmethod
    def create_sql_repository(session=None) -> SQLDatabaseRepository:
        """Create SQL repository explicitly"""
        return SQLDatabaseRepository(session=session)
    
    @staticmethod
    def create_mongo_repository(connection_string: str = None, 
                                database_name: str = None) -> MongoDatabaseRepository:
        """Create MongoDB repository explicitly"""
        return MongoDatabaseRepository(
            connection_string=connection_string,
            database_name=database_name or 'crypto_prediction'
        )


# Singleton instance for DI container
_db_repository: Optional[object] = None


def get_database_repository() -> object:
    """Get database repository instance (singleton)"""
    global _db_repository
    if _db_repository is None:
        _db_repository = DatabaseFactory.create_repository()
    return _db_repository


def reset_database_repository():
    """Reset database repository (for testing)"""
    global _db_repository
    if _db_repository is not None:
        _db_repository.close()
        _db_repository = None
