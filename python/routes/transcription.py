from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File
from utils.audio_processor import clean_audio_for_diarization
# from utils.deepgram_processor import transcribe_with_diarization, build_raw_transcript  # Deepgram
from utils.sarvam_processor import transcribe_with_sarvam  # Sarvam AI
from utils.deepgram_processor import build_raw_transcript  # Still need this for speaker filtering
from utils.gemini_processor import refine_transcript
from utils.embedding_processor import generate_embedding
from database import db

router = APIRouter(redirect_slashes=False)

# MongoDB collection
transcriptions = db["transcriptions"]


@router.post("")
@router.post("/")
async def save_transcription(call_recording: UploadFile = File(...)):
    try:
        audio_bytes = await call_recording.read()

        print(f"\n{'═' * 60}")
        print(f"📞 Processing audio: {call_recording.filename} ({len(audio_bytes) / 1024:.1f} KB)")
        print(f"{'═' * 60}")

        # STEP 1: ML-based audio cleaning (noisereduce + compression)
        cleaned_bytes = clean_audio_for_diarization(audio_bytes)

        # STEP 2: Sarvam AI transcription + diarization (optimized for Indian languages)
        dg_result = await transcribe_with_sarvam(cleaned_bytes, call_recording.filename)

        if not dg_result["has_speech"]:
            return {
                "message": "Transcription complete but no speech/speakers detected.",
                "transcript": dg_result["transcript"]
            }

        # STEP 3: Build raw transcript with speaker filtering
        raw_data = build_raw_transcript(dg_result["words"])

        # STEP 4: LangChain + Gemini refinement
        print("🧠 Sending to Gemini for full technical analysis & refinement...")
        gemini_result = await refine_transcript(
            raw_data["raw_transcript"],
            len(raw_data["valid_speakers"])
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
                "processingMs": {
                    "stt": dg_result["processing_time"],
                    "gemini": gemini_result["processing_time"],
                },
            },
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
        print(f"   Confidence: {diarization_confidence * 100:.1f}%")
        print(f"   Embedding: {len(embedding)} dims")
        print(f"   MongoDB ID: {doc_id}")
        print(f"{'─' * 60}\n")

        # Return same JSON as before + the MongoDB doc ID
        return {
            "success": True,
            "_id": doc_id,
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
