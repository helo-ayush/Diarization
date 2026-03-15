# ==============================================================================
# AGENT ENROLLMENT ROUTE
# This file handles registering new agents by calculating their voice print.
# ==============================================================================
import os
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from utils.fingerprint_processor import extract_voice_embedding
from database import db

router = APIRouter(redirect_slashes=False)

# MongoDB collection
agents_collection = db["agents"]

@router.post("/enroll")
async def enroll_agent(
    name: str = Form(...),
    voice_sample: UploadFile = File(...)
):
    """
    Enrolls a new agent by extracting an audio sample and generating a 
    256-dimensional Voice Print (embedding) via Pyannote. 
    This print is saved to MongoDB to be used later for agent identification.
    """
    try:
        # Securely read file into memory
        audio_bytes = await voice_sample.read()
        
        print(f"\n{'═' * 60}")
        print(f"👤 Enrolling Agent: {name}")
        print(f"🎙️ Audio size: {len(audio_bytes) / 1024:.1f} KB")
        print(f"{'═' * 60}")

        # STEP 1: Pass the raw audio into Pyannote and receive a math vector
        embedding = extract_voice_embedding(audio_bytes, voice_sample.filename)
        print(f"   🧬 Voice embedding extracted: {len(embedding)} dimensions")

        # STEP 2: Package the metadata and vector for storage
        agent_doc = {
            "name": name,
            "voice_embedding": embedding,
            "updatedAt": datetime.now(timezone.utc),
        }
        
        # STEP 3: Save to Database (Upsert prevents duplicates based on name)
        agent_id = name.lower().replace(" ", "_").strip()
        await agents_collection.update_one(
            {"_id": agent_id},
            {"$set": agent_doc},
            upsert=True
        )

        print(f"✅ Agent '{name}' enrolled successfully.")

        return {
            "success": True,
            "message": f"Successfully enrolled agent: {name}",
            "agent_id": agent_id
        }

    except Exception as e:
        print(f"❌ Error enrolling agent: {e}")
        # Standardize 500 error on internal failure
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_agents():
    """
    Returns a list of all currently enrolled agents.
    Excludes the heavy 'voice_embedding' array from the output to save bandwidth.
    """
    agents = await agents_collection.find({}, {"voice_embedding": 0}).to_list(length=100)
    return {"agents": agents}
