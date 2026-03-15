import os
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from utils.fingerprint_processor import extract_voice_embedding
from database import db

router = APIRouter(redirect_slashes=False)
agents_collection = db["agents"]

@router.post("/enroll")
async def enroll_agent(
    name: str = Form(...),
    voice_sample: UploadFile = File(...)
):
    """
    Enrolls a new agent by saving their generated Voice Fingerprint to MongoDB.
    """
    try:
        audio_bytes = await voice_sample.read()
        
        print(f"\n{'═' * 60}")
        print(f"👤 Enrolling Agent: {name}")
        print(f"🎙️ Audio size: {len(audio_bytes) / 1024:.1f} KB")
        print(f"{'═' * 60}")

        # 1. Extract 192-dimensional Voice Print
        embedding = extract_voice_embedding(audio_bytes, voice_sample.filename)
        print(f"   🧬 Voice embedding extracted: {len(embedding)} dimensions")

        # 2. Save/Update in MongoDB
        agent_doc = {
            "name": name,
            "voice_embedding": embedding,
            "updatedAt": datetime.now(timezone.utc),
        }
        
        # Use simple identifier for the agent_id based on name
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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_agents():
    """Returns a list of all enrolled agents."""
    agents = await agents_collection.find({}, {"voice_embedding": 0}).to_list(length=100)
    return {"agents": agents}
