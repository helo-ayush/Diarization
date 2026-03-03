import os
import time
import asyncio
from deepgram import DeepgramClient, PrerecordedOptions


deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))


async def transcribe_with_diarization(audio_bytes: bytes) -> dict:
    """
    Transcribe audio with speaker diarization using Deepgram Nova-3.
    Returns word-level data with speaker assignments.
    """
    size_kb = len(audio_bytes) / 1024
    print(f"🎙️ Sending {size_kb:.1f}KB to Deepgram...")

    start_time = time.time()

    options = PrerecordedOptions(
        model="nova-3",
        language="multi",
        diarize=True,
        smart_format=True,
        punctuate=True,
        utterances=True,
        multichannel=False,
    )

    source = {"buffer": audio_bytes}

    # Run with 120s timeout
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                deepgram.listen.rest.v("1").transcribe_file,
                source,
                options,
            ),
            timeout=120
        )
    except asyncio.TimeoutError:
        raise TimeoutError("Deepgram timed out after 120s")

    processing_time = int((time.time() - start_time) * 1000)
    print(f"✅ Deepgram completed in {processing_time}ms")

    utterances = response.results.utterances
    channel = response.results.channels[0] if response.results.channels else None
    words_raw = channel.alternatives[0].words if channel and channel.alternatives else []

    # No speech detected
    if not utterances or len(utterances) == 0:
        transcript = channel.alternatives[0].transcript if channel and channel.alternatives else ""
        return {
            "words": [],
            "transcript": transcript,
            "has_speech": False,
            "processing_time": processing_time
        }

    # Build word-level array with speaker info
    words = [
        {
            "word": w.punctuated_word or w.word,
            "speaker": w.speaker,
            "confidence": getattr(w, "speaker_confidence", None) or w.confidence or 0,
            "start": w.start,
            "end": w.end,
        }
        for w in words_raw
    ]

    return {"words": words, "has_speech": True, "processing_time": processing_time}


def build_raw_transcript(words: list) -> dict:
    """
    Filter out background noise speakers (< 5% word share)
    and build a raw transcript string from valid words.
    """
    total_words = len(words)

    # Count words per speaker
    speaker_counts = {}
    for w in words:
        speaker_counts[w["speaker"]] = speaker_counts.get(w["speaker"], 0) + 1

    # Filter speakers with < 5% word share
    valid_speakers = [
        int(s) for s, count in speaker_counts.items()
        if (count / total_words) > 0.05
    ]

    print(f"👥 Speakers detected: {len(speaker_counts)}, valid: {len(valid_speakers)}")
    for spk, count in speaker_counts.items():
        pct = (count / total_words) * 100
        valid = int(spk) in valid_speakers
        print(f"   Speaker {spk}: {count} words ({pct:.1f}%) {'✅' if valid else '❌ filtered'}")

    # Filter words and group into speaker segments
    valid_words = [w for w in words if w["speaker"] in valid_speakers]
    segments = []
    current_seg = None

    for w in valid_words:
        if current_seg is None or current_seg["speaker"] != w["speaker"]:
            if current_seg:
                segments.append(current_seg)
            current_seg = {"speaker": w["speaker"], "words": [w["word"]]}
        else:
            current_seg["words"].append(w["word"])
    if current_seg:
        segments.append(current_seg)

    raw_transcript = "\n".join(
        f"Speaker {s['speaker']}: {' '.join(s['words'])}" for s in segments
    )

    # Confidence stats
    avg_confidence = sum(w["confidence"] for w in valid_words) / len(valid_words) if valid_words else 0
    low_conf_words = sum(1 for w in valid_words if w["confidence"] < 0.7)
    print(f"📊 Deepgram confidence: avg={avg_confidence:.3f}, low-conf words={low_conf_words}/{len(valid_words)}")

    return {
        "raw_transcript": raw_transcript,
        "valid_words": valid_words,
        "valid_speakers": valid_speakers,
        "segments": segments,
        "avg_confidence": avg_confidence,
        "low_conf_words": low_conf_words,
    }
