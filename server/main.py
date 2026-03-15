import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import mongo_client
from routes.transcription import router as transcription_router
from routes.search import router as search_router
from routes.agents import router as agents_router

# ==============================================================================
# APP LIFECYCLE MANAGEMENT
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of the FastAPI application.
    Executes startup scripts (like database connections) before accepting requests,
    and cleanup scripts when the server stops.
    """
    # Startup: Verify database connection
    try:
        await mongo_client.admin.command("ping")
        print("✅ Connected to MongoDB")
    except Exception as e:
        print(f"❌ MongoDB Connection Error: {e}")
    yield
    # Shutdown: Close database connections gracefully to prevent memory leaks
    mongo_client.close()

# Initialize FastAPI application
app = FastAPI(title="Voice Diarization API", lifespan=lifespan)

# ==============================================================================
# SECURITY & CORS CONFIGURATION
# ==============================================================================
# Allows the React frontend to communicate with this backend API securely.
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
origins = [
    frontend_url,
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# ROUTE REGISTRATION
# ==============================================================================
@app.get("/")
async def health():
    """Health check endpoint to verify the server is running."""
    return {"status": "Voice Diarization API is running."}

# Register individual feature routers with a consistent API v1 prefix
app.include_router(transcription_router, prefix="/api/v1/transcriptions")
app.include_router(search_router, prefix="/api/v1/search")
app.include_router(agents_router, prefix="/api/v1/agents")
