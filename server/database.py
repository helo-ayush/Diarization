import os
from motor.motor_asyncio import AsyncIOMotorClient

# ==============================================================================
# DATABASE CONFIGURATION
# ==============================================================================
# Initializes an asynchronous MongoDB client using the Motor library.
# The URI is securely fetched from the environment variables.
mongo_client = AsyncIOMotorClient(os.getenv("MONGO_URI"))

# Select the primary database instance
db = mongo_client["diarization"]
