import os
import time
import tempfile
from sarvamai import SarvamAI

client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))


async def transcribe_with_sarvam(audio_bytes: bytes, filename: str = "audio.ogg") -> dict:
    """
    Transcribe audio with speaker diarization using Sarvam AI's Batch API.
    
    Uses the saaras:v3 model optimized for Indian languages (Hindi, Hinglish, etc.)
    with speaker diarization enabled.
    """
    size_kb = len(audio_bytes) / 1024
    print(f"🎙️ Sending {size_kb:.1f}KB to Sarvam AI...")

    start_time = time.time()

    # Save audio to temp file (Sarvam SDK needs file paths)
    with tempfile.NamedTemporaryFile(suffix=f".{filename.split('.')[-1]}", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # Create batch job with diarization
        print("   📋 Creating Sarvam batch job...")
        job = client.speech_to_text_job.create_job(
            model="saaras:v3",
            mode="transcribe",
            language_code="hi-IN",
            with_diarization=True,
            num_speakers=2,
        )
        print(f"   ✅ Job created: {job.id if hasattr(job, 'id') else 'OK'}")

        # Upload audio file
        print("   📤 Uploading audio file...")
        job.upload_files(file_paths=[tmp_path])

        # Start processing
        print("   ⏳ Processing started...")
        job.start()

        # Wait for completion
        print("   ⏳ Waiting for completion...")
        job.wait_until_complete()

        processing_time = int((time.time() - start_time) * 1000)
        print(f"   ✅ Sarvam completed in {processing_time}ms")

        # Get results
        file_results = job.get_file_results()
        successful = file_results.get("successful", [])

        if not successful:
            failed = file_results.get("failed", [])
            error_msg = failed[0].get("error_message", "Unknown error") if failed else "No results"
            print(f"   ❌ Sarvam transcription failed: {error_msg}")
            return {
                "words": [],
                "transcript": "",
                "has_speech": False,
                "processing_time": processing_time,
            }

        # Download and parse the output
        output_dir = tempfile.mkdtemp()
        job.download_outputs(output_dir=output_dir)

        # Read the output JSON file
        import json
        output_files = os.listdir(output_dir)
        result_data = None
        for f in output_files:
            if f.endswith(".json"):
                with open(os.path.join(output_dir, f), encoding="utf-8") as fp:
                    result_data = json.load(fp)
                break

        if not result_data:
            print("   ❌ No output file found")
            return {
                "words": [],
                "transcript": "",
                "has_speech": False,
                "processing_time": processing_time,
            }

        # Extract diarized transcript
        diarized = result_data.get("diarized_transcript", {})
        entries = diarized.get("entries", [])

        if not entries:
            # Fallback to plain transcript
            plain = result_data.get("transcript", "")
            return {
                "words": [],
                "transcript": plain,
                "has_speech": bool(plain),
                "processing_time": processing_time,
            }

        # Build word-level-like data from diarized entries
        # Sarvam returns sentence-level segments, not word-level
        words = []
        for entry in entries:
            text = entry.get("transcript", "")
            speaker = int(entry.get("speaker_id", 0))
            start = entry.get("start_time_seconds", 0)
            end = entry.get("end_time_seconds", 0)

            # Split into words for compatibility with our pipeline
            for word in text.split():
                words.append({
                    "word": word,
                    "speaker": speaker,
                    "confidence": 0.8,  # Sarvam doesn't provide word-level confidence
                    "start": start,
                    "end": end,
                })

        print(f"   📊 Sarvam: {len(entries)} segments, {len(words)} words")
        return {"words": words, "has_speech": True, "processing_time": processing_time}

    finally:
        # Cleanup
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
