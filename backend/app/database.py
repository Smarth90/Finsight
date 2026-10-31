from pymongo import MongoClient
from pymongo.server_api import ServerApi

from app.config import MONGODB_URL, DATABASE_NAME

if not MONGODB_URL:
    raise ValueError("MONGODB_URL is missing")

client = MongoClient(
    MONGODB_URL,
    server_api=ServerApi("1")
)

database = client[DATABASE_NAME]

def test_connection():
    try:
        client.admin.command("ping")
        print("MongoDB connection successful!")
        print(f"Connected database: {DATABASE_NAME}")
        return True

    except Exception as error:
        print(f"MongoDB connection failed: {error}")
        return False