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

print("⏳ Loading Silero VAD model (downloading if first time)...")
vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", 
    model="silero_vad", 
    force_reload=False, 
    onnx=False
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = vad_utils
print("✅ Silero VAD loaded.")

def clean_audio_for_diarization(audio_bytes: bytes, original_filename: str = "input.wav") -> bytes:
    """
    Clean audio using ML-based VAD (Silero) + Noise Reduction (noisereduce).
    
    Pipeline:
    1. ffmpeg converts any input format → WAV (16kHz, mono) in memory
    2. Silero VAD detects and extracts only speech segments (ignoring noise/silence)
    3. noisereduce applies ML spectral gating noise reduction
    4. Highpass/lowpass filtering via FFT
    5. ffmpeg compresses result → OGG/Opus for fast upload
    """
    print(f"🔧 Cleaning audio {original_filename} with ML noise reduction & VAD...")

    # Step 1: Convert to WAV 16kHz mono using ffmpeg
    ext = os.path.splitext(original_filename)[1] or ".input"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name

    tmp_wav_path = tmp_in_path + ".wav"

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_in_path,
                "-ac", "1",           # mono
                "-ar", "16000",       # 16kHz
                "-sample_fmt", "s16", # 16-bit
                tmp_wav_path
            ],
            capture_output=True,
            check=True,
        )

        # Step 2: Load WAV into numpy
        samples, sample_rate = sf.read(tmp_wav_path, dtype="float32")

        # Step 3: ML Voice Activity Detection (Silero VAD)
        # Accurately snips out all non-speech audio
        tensor_samples = torch.from_numpy(samples)
        
        # We configure VAD to be more forgiving for short utterances (like numbers)
        # and add padding to prevent abrupt clipping.
        speech_timestamps = get_speech_timestamps(
            tensor_samples, 
            vad_model, 
            sampling_rate=sample_rate,
            threshold=0.3,                 # Lower threshold to catch quieter speech
            min_speech_duration_ms=100,    # Lower minimum duration for short words (like "one")
            min_silence_duration_ms=700,   # If silence is < 700ms, don't split the audio. This keeps sentences natural.
            speech_pad_ms=250              # Add 250ms of padding to the start and end of every speech chunk so it doesn't sound clipped
        )
        
        if not speech_timestamps:
            print("⚠️ No speech detected using Silero VAD. Using original audio...")
            speech_samples = samples
        else:
            vad_tensor = collect_chunks(speech_timestamps, tensor_samples)
            speech_samples = vad_tensor.numpy()
            print(f"✂️ VAD trimming: {(len(samples)/16000):.1f}s → {(len(speech_samples)/16000):.1f}s")

        # Step 4: ML Noise Reduction
        cleaned = nr.reduce_noise(
            y=speech_samples,
            sr=sample_rate,
            prop_decrease=0.85,       # Remove 85% of noise
            stationary=False,         # Handle non-stationary noise (traffic, wind)
            n_fft=2048,
            freq_mask_smooth_hz=500,
        )

        # Step 4: Highpass (80Hz) + Lowpass (8kHz) via FFT
        fft = np.fft.rfft(cleaned)
        freqs = np.fft.rfftfreq(len(cleaned), d=1/sample_rate)
        fft[freqs < 80] = 0
        fft[freqs > 8000] = 0
        cleaned = np.fft.irfft(fft, n=len(cleaned))

        # Normalize
        peak = np.max(np.abs(cleaned))
        if peak > 0:
            cleaned = cleaned / peak

        # Step 5: Write cleaned WAV
        tmp_clean_path = tmp_in_path + ".clean.wav"
        sf.write(tmp_clean_path, cleaned, sample_rate, subtype="PCM_16")

        # Step 6: Compress to OGG/Opus
        tmp_ogg_path = tmp_in_path + ".ogg"
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_clean_path,
                "-c:a", "libopus",
                "-b:a", "32k",
                tmp_ogg_path
            ],
            capture_output=True,
            check=True,
        )

        with open(tmp_ogg_path, "rb") as f:
            ogg_bytes = f.read()

        original_kb = len(audio_bytes) / 1024
        cleaned_kb = len(ogg_bytes) / 1024
        print(f"✅ Audio cleaned: {original_kb:.0f}KB → {cleaned_kb:.0f}KB (ML denoised + compressed)")

        return ogg_bytes

    finally:
        # Cleanup temp files
        for path in [tmp_in_path, tmp_wav_path, tmp_clean_path, tmp_ogg_path]:
            try:
                os.unlink(path)
            except (OSError, UnboundLocalError):
                pass
