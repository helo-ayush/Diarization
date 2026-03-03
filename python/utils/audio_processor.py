import io
import subprocess
import tempfile
import os
import numpy as np
import soundfile as sf
import noisereduce as nr


def clean_audio_for_diarization(audio_bytes: bytes, original_filename: str = "input.wav") -> bytes:
    """
    Clean audio using ML-based noise reduction (noisereduce) + basic filtering.
    
    Pipeline:
    1. ffmpeg converts any input format → WAV (16kHz, mono) in memory
    2. noisereduce applies ML spectral gating noise reduction
    3. Highpass/lowpass filtering via FFT
    4. ffmpeg compresses result → OGG/Opus for fast upload
    
    Returns: OGG/Opus compressed bytes ready for Deepgram
    """
    print(f"🔧 Cleaning audio {original_filename} with ML noise reduction...")

    # Step 1: Convert any format to WAV 16kHz mono using ffmpeg
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

        # Step 3: ML Noise Reduction (the big upgrade over ffmpeg's afftdn)
        # noisereduce uses spectral gating: it learns the noise profile
        # and subtracts it, preserving speech characteristics.
        cleaned = nr.reduce_noise(
            y=samples,
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
