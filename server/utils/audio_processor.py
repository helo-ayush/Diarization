# ==============================================================================
# AUDIO PROCESSOR
# Handles noise reduction, VAD (Voice Activity Detection), and audio conversion.
# This prevents garbage audio from being sent to transcription APIs.
#
# FIX: Previously used collect_chunks() which hard-concatenated speech segments
# with zero gap between them, causing acoustic discontinuity that confused
# Sarvam's language model context window. Now replaces removed silence with a
# short 300ms synthetic pad + 10ms fade in/out at each boundary.
# ==============================================================================
import io
import subprocess
import tempfile
import os
import numpy as np
import soundfile as sf
import noisereduce as nr

import torch
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# VAD CONFIGURATION
# ==============================================================================
# How long of a silence gap to insert between joined speech chunks.
# 300ms sounds like a natural short pause — long enough for Sarvam's context
# window to register a speaker breath, short enough to still reduce file size.
SILENCE_PAD_BETWEEN_CHUNKS_S = 0.3

# How many samples to use for fade in/out at each chunk boundary.
# 10ms prevents amplitude clicking at hard splice points.
FADE_DURATION_S = 0.01

print("⏳ Loading Silero VAD model (downloading if first time)...")
vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = vad_utils
print("✅ Silero VAD loaded.")


# ==============================================================================
# INTERNAL HELPERS
# ==============================================================================

def _apply_fade(chunk: np.ndarray, fade_samples: int) -> np.ndarray:
    """
    Apply a short linear fade-in at the start and fade-out at the end of a
    chunk. Removes the audible click/pop that occurs when two audio segments
    are hard-spliced together at different amplitude levels.
    """
    if len(chunk) < fade_samples * 2:
        # Chunk is too short to safely apply fade without overlap — skip it
        return chunk

    c = chunk.copy()
    c[:fade_samples]  *= np.linspace(0.0, 1.0, fade_samples)   # fade in
    c[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)   # fade out
    return c


def _join_speech_chunks(
    speech_timestamps: list,
    tensor_samples: torch.Tensor,
    sample_rate: int,
) -> np.ndarray:
    """
    Joins speech segments from VAD with a short silence pad between each one,
    instead of hard-concatenating them with zero gap (old collect_chunks behavior).

    This preserves natural acoustic pacing so Sarvam's language model doesn't
    see abrupt teleporting between unrelated audio frames.
    """
    fade_samples = int(FADE_DURATION_S * sample_rate)
    silence_pad  = np.zeros(
        int(SILENCE_PAD_BETWEEN_CHUNKS_S * sample_rate),
        dtype=np.float32
    )

    chunks = []
    for i, ts in enumerate(speech_timestamps):
        # Extract this speech segment as a numpy array
        chunk = tensor_samples[ts["start"] : ts["end"]].numpy()

        # Apply fade in/out to smooth the amplitude at splice boundaries
        chunk = _apply_fade(chunk, fade_samples)

        chunks.append(chunk)

        # Insert silence gap between chunks (not after the last one)
        if i < len(speech_timestamps) - 1:
            chunks.append(silence_pad)

    return np.concatenate(chunks)


# ==============================================================================
# MAIN PUBLIC FUNCTION
# ==============================================================================

def clean_audio_for_diarization(audio_bytes: bytes, original_filename: str = "input.wav") -> bytes:
    """
    Clean audio using ML-based VAD (Silero) + Noise Reduction (noisereduce).

    Pipeline:
    1. ffmpeg converts any input format → WAV (16kHz, mono) in memory
    2. Silero VAD detects speech segments
    3. Speech segments are joined with 300ms silence pads + 10ms fades
       (replaces the old hard-concat collect_chunks approach)
    4. noisereduce applies ML spectral gating noise reduction
    5. Highpass (80Hz) + Lowpass (8kHz) filtering via FFT
    6. ffmpeg compresses result → OGG/Opus for fast upload
    """
    print(f"🔧 Cleaning audio '{original_filename}' with ML noise reduction & VAD...")

    # ── Step 1: Convert input to WAV 16kHz mono via ffmpeg ──────────────────
    ext = os.path.splitext(original_filename)[1] or ".input"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name

    tmp_wav_path   = tmp_in_path + ".wav"
    tmp_clean_path = tmp_in_path + ".clean.wav"
    tmp_ogg_path   = tmp_in_path + ".ogg"

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_in_path,
                "-ac", "1",            # mono
                "-ar", "16000",        # 16 kHz
                "-sample_fmt", "s16",  # 16-bit PCM
                tmp_wav_path,
            ],
            capture_output=True,
            check=True,
        )

        # ── Step 2: Load WAV into numpy ──────────────────────────────────────
        samples, sample_rate = sf.read(tmp_wav_path, dtype="float32")
        original_dur = len(samples) / sample_rate

        # ── Step 3: Silero VAD — detect speech timestamps ───────────────────
        tensor_samples = torch.from_numpy(samples)

        speech_timestamps = get_speech_timestamps(
            tensor_samples,
            vad_model,
            sampling_rate=sample_rate,
            threshold=0.3,                # Lower = catch quieter speech
            min_speech_duration_ms=100,   # Catch short words like "haan", "ok"
            min_silence_duration_ms=700,  # Don't split on pauses < 700ms
            speech_pad_ms=250,            # 250ms padding around each chunk
        )

        if not speech_timestamps:
            print("⚠️  No speech detected by Silero VAD — using original audio.")
            speech_samples = samples
        else:
            # ── Step 3b: Join chunks with silence pads (THE KEY FIX) ────────
            speech_samples = _join_speech_chunks(
                speech_timestamps,
                tensor_samples,
                sample_rate,
            )
            new_dur = len(speech_samples) / sample_rate
            saved   = original_dur - new_dur
            print(
                f"✂️  VAD trimming: {original_dur:.1f}s → {new_dur:.1f}s "
                f"({saved:.1f}s of silence removed, "
                f"{len(speech_timestamps)} chunks joined with {SILENCE_PAD_BETWEEN_CHUNKS_S*1000:.0f}ms pads)"
            )

        # ── Step 4: ML Noise Reduction ───────────────────────────────────────
        cleaned = nr.reduce_noise(
            y=speech_samples,
            sr=sample_rate,
            prop_decrease=0.85,      # Remove 85% of noise
            stationary=False,        # Handle non-stationary noise (traffic, fans)
            n_fft=2048,
            freq_mask_smooth_hz=500,
        )

        # ── Step 5: Highpass (80Hz) + Lowpass (8kHz) via FFT ─────────────────
        fft   = np.fft.rfft(cleaned)
        freqs = np.fft.rfftfreq(len(cleaned), d=1 / sample_rate)
        fft[freqs < 80]   = 0   # cut sub-bass rumble
        fft[freqs > 8000] = 0   # cut high-frequency hiss
        cleaned = np.fft.irfft(fft, n=len(cleaned))

        # Normalize to prevent clipping
        peak = np.max(np.abs(cleaned))
        if peak > 0:
            cleaned = cleaned / peak

        # ── Step 6: Write cleaned WAV ─────────────────────────────────────────
        sf.write(tmp_clean_path, cleaned, sample_rate, subtype="PCM_16")

        # ── Step 7: Compress to OGG/Opus ─────────────────────────────────────
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_clean_path,
                "-c:a", "libopus",
                "-b:a", "32k",
                tmp_ogg_path,
            ],
            capture_output=True,
            check=True,
        )

        with open(tmp_ogg_path, "rb") as f:
            ogg_bytes = f.read()

        original_kb = len(audio_bytes) / 1024
        cleaned_kb  = len(ogg_bytes)   / 1024
        print(
            f"✅ Audio cleaned: {original_kb:.0f} KB → {cleaned_kb:.0f} KB "
            f"(ML denoised + VAD-trimmed + compressed)"
        )

        return ogg_bytes

    finally:
        # Always clean up temp files even if an exception occurred
        for path in [tmp_in_path, tmp_wav_path, tmp_clean_path, tmp_ogg_path]:
            try:
                os.unlink(path)
            except (OSError, UnboundLocalError):
                pass