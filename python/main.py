import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import mongo_client
from routes.transcription import router as transcription_router
from routes.search import router as search_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        await mongo_client.admin.command("ping")
        print("✅ Connected to MongoDB")
    except Exception as e:
        print(f"❌ MongoDB Connection Error: {e}")
    yield
    # Shutdown
    mongo_client.close()


app = FastAPI(title="Voice Diarization API", lifespan=lifespan)

# CORS
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


@app.get("/")
async def health():
    return {"status": "Voice Diarization API is running."}


# API Routes
app.include_router(transcription_router, prefix="/save")
app.include_router(search_router, prefix="/search")
