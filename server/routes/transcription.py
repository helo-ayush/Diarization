# ==============================================================================
# TRANSCRIPTION ROUTE
# This file serves as the main orchestrator for the audio processing pipeline.
# ==============================================================================
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File
from utils.audio_processor import clean_audio_for_diarization
from utils.sarvam_processor import transcribe_with_sarvam  # Sarvam AI for transcription
from utils.deepgram_processor import build_raw_transcript  # Utility for parsing raw segment data
from utils.gemini_processor import refine_transcript       # Gemini for sentiment analysis
from utils.embedding_processor import generate_embedding   # Embeddings for RAG search
from utils.fingerprint_processor import extract_voice_embedding # Pyannote Voice Fingerprinting
from database import db

router = APIRouter(redirect_slashes=False)

# ==============================================================================
# DATABASE COLLECTIONS
# ==============================================================================
transcriptions = db["transcriptions"]
agents_collection = db["agents"]

def cosine_similarity(v1, v2):
    """
    Computes the mathematical similarity between two embedding vectors.
    Returns a score between 0 and 1, where 1 means exact match.
    """
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)

async def identify_agents_in_audio(audio_bytes: bytes, filename: str, segments: list, valid_speakers: list) -> dict:
    """
    Iterates through each valid speaker detected by the transcription service,
    extracts an audio snippet of their voice, and compares it to enrolled agents 
    in the database via Cosine Similarity.
    
    Returns a dictionary mapping speaker IDs to Agent Names (e.g. {"0": "Ayush"})
    """
    agents = await agents_collection.find({}).to_list(length=100)
    if not agents:
        return {}

    agent_map = {}
    print("🔍 Attempting Pyannote Voice Fingerprinting for Agent Identification...")
    for spk in valid_speakers:
        # Find the longest continuous segment for this speaker to get a clean voice sample
        spk_segments = [s for s in segments if s["speaker"] == spk]
        if not spk_segments:
            continue
            
        longest_seg = max(spk_segments, key=lambda s: s["end"] - s["start"])
        duration = longest_seg["end"] - longest_seg["start"]
        
        # Audio snippets under 1 second yield poor embeddings.
        if duration < 1.0:
            print(f"   ⚠️ Speaker {spk} longest segment is too short ({duration:.1f}s). Skipping.")
            continue
            
        print(f"   🎧 Extracting embedding for Speaker {spk} ({longest_seg['start']:.1f}s - {longest_seg['end']:.1f}s)...")
        
        try:
            # Add a 0.5s padding to start/end to gather better context for the neural network
            start_time = max(0.0, longest_seg["start"] - 0.5)
            end_time = longest_seg["end"] + 0.5
            
            # Generate the 256-dimensional Voice Print
            emb = extract_voice_embedding(audio_bytes, filename, start_sec=start_time, end_sec=end_time)
            
            # Compare the generated embedding against all known agents in MongoDB
            best_match = None
            best_score = -1
            
            for agent in agents:
                score = cosine_similarity(emb, agent["voice_embedding"])
                if score > best_score:
                    best_score = score
                    best_match = agent["name"]
            
            # 0.45 is typically a confident threshold for the ResNet34 architecture
            # If matched, we record it into our agent map.
            if best_score > 0.45:
                agent_map[spk] = best_match
                print(f"   ✅ Speaker {spk} matched as {best_match} (Score: {best_score:.3f})")
            else:
                print(f"   ❌ Speaker {spk} did not match any agent (Best score: {best_score:.3f} for {best_match})")
                
        except Exception as e:
            print(f"   ⚠️ Error extracting embedding for Speaker {spk}: {e}")

    return agent_map

# ==============================================================================
# MAIN PIPELINE ENDPOINT
# ==============================================================================
@router.post("")
@router.post("/")
async def save_transcription(call_recording: UploadFile = File(...)):
    """
    Receives an audio file, transcribes it, identifies the agent via voice 
    fingerprinting, generates sentiment analysis via Gemini, and stores the 
    result + vector embeddings into MongoDB for later search.
    """
        audio_bytes = await call_recording.read()

        print(f"\n{'═' * 60}")
        print(f"📞 Processing audio: {call_recording.filename} ({len(audio_bytes) / 1024:.1f} KB)")
        print(f"{'═' * 60}")

        # STEP 1: ML-based audio cleaning (noisereduce + compression)
        cleaned_bytes = clean_audio_for_diarization(audio_bytes, call_recording.filename)

        # STEP 2: Sarvam AI transcription + diarization (optimized for Indian languages)
        dg_result = await transcribe_with_sarvam(cleaned_bytes, call_recording.filename)

        if not dg_result["has_speech"]:
            return {
                "message": "Transcription complete but no speech/speakers detected.",
                "transcript": dg_result["transcript"]
            }

        # STEP 3: Build raw transcript with speaker filtering
        raw_data = build_raw_transcript(dg_result["words"])
        
        # STEP 3.5: Identify Agents via Voice Fingerprinting
        agent_map = await identify_agents_in_audio(cleaned_bytes, call_recording.filename, raw_data["segments"], raw_data["valid_speakers"])
        
        # Re-build raw transcript if any agents were found
        pynnote_diarized = False
        if agent_map:
            pynnote_diarized = True
            raw_data = build_raw_transcript(dg_result["words"], agent_map=agent_map)

        # STEP 4: LangChain + Gemini refinement
        print("🧠 Sending to Gemini for full technical analysis & refinement...")
        gemini_result = await refine_transcript(
            raw_data["raw_transcript"],
            len(raw_data["valid_speakers"]),
            pynnote_diarized=pynnote_diarized
        )
        print(f"✅ Gemini refinement completed in {gemini_result['processing_time']}ms")

        # Fallback logic
        final_transcript = gemini_result["transcript"] or raw_data["raw_transcript"]
        was_refined = gemini_result["transcript"] is not None
        valid_count = len(raw_data["valid_words"])
        diarization_confidence = 1 - (raw_data["low_conf_words"] / valid_count) if valid_count else 0

        # STEP 5: Generate embedding from summary for vector search
        summary = gemini_result["summary"] or ""
        embedding = await generate_embedding(summary) if summary else []

        # STEP 6: Save to MongoDB
        mongo_doc = {
            "filename": call_recording.filename,
            "transcript": final_transcript,
            "summary": summary,
            "satisfactionScore": gemini_result["satisfaction_score"],
            "tags": gemini_result["tags"],
            "detectedRoles": gemini_result.get("detected_roles", {}),
            "speakerCount": len(raw_data["valid_speakers"]),
            "metrics": {
                "totalWords": valid_count,
                "rawSegments": len(raw_data["segments"]),
                "geminiRefined": was_refined,
                "diarizationConfidence": round(diarization_confidence, 3),
                "pynoteDiarized": pynnote_diarized,
                "processingMs": {
                    "stt": dg_result["processing_time"],
                    "gemini": gemini_result["processing_time"],
                },
            },
            "pynoteDiarized": pynnote_diarized,
            "embedding": embedding,
            "createdAt": datetime.now(timezone.utc),
        }

        result = await transcriptions.insert_one(mongo_doc)
        doc_id = str(result.inserted_id)
        print(f"💾 Saved to MongoDB: {doc_id}")

        # Summary log
        print(f"\n{'─' * 60}")
        print(f"📋 ANALYSIS RESULTS:")
        print(f"   Words: {valid_count} | Speakers: {len(raw_data['valid_speakers'])}")
        print(f"   Satisfaction Score: {gemini_result['satisfaction_score']}/10")
        print(f"   Tags: [{', '.join(gemini_result['tags'])}]")
        print(f"   Gemini refined: {'✅ YES' if was_refined else '❌ NO'}")
        print(f"   Pyannote matched: {'✅ YES' if pynnote_diarized else '❌ NO'}")
        print(f"   Confidence: {diarization_confidence * 100:.1f}%")
        print(f"   Embedding: {len(embedding)} dims")
        print(f"   MongoDB ID: {doc_id}")
        print(f"{'─' * 60}\n")

        # Return same JSON as before + the MongoDB doc ID
        return {
            "success": True,
            "_id": doc_id,
            "pynoteDiarized": pynnote_diarized,
            "analysis": {
                "summary": summary,
                "satisfactionScore": gemini_result["satisfaction_score"],
                "tags": gemini_result["tags"],
                "detectedRoles": gemini_result.get("detected_roles", {}),
            },
            "transcript": final_transcript,
            "speakerCount": len(raw_data["valid_speakers"]),
            "metrics": {
                "totalWords": valid_count,
                "rawSegments": len(raw_data["segments"]),
                "geminiRefined": was_refined,
                "diarizationConfidence": round(diarization_confidence, 3),
                "pynoteDiarized": pynnote_diarized,
                "processingMs": {
                    "stt": dg_result["processing_time"],
                    "gemini": gemini_result["processing_time"],
                },
            },
        }

    except Exception as e:
        print(f"Transcription Route Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to process audio transcription: {str(e)}"}
