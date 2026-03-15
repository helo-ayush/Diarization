import io
import os
import tempfile
import torch
import torchaudio
import subprocess

print("⏳ Loading Pyannote Voice Fingerprinting model...")

_model = None

def _get_model():
    global _model
    if _model is None:
        from pyannote.audio import Model
        from pyannote.audio import Inference
        import torch

        import torchaudio
        import soundfile as sf
        import warnings

        # PyTorch 2.6+ defaults to weights_only=True, which breaks Pyannote.
        # We temporarily patch torch.load to use weights_only=False.
        _orig_load = torch.load
        def _safe_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return _orig_load(*args, **kwargs)
        
        torch.load = _safe_load

        # Patch torchaudio.load to use soundfile instead of torchcodec (which fails on Windows)
        if not hasattr(torchaudio, "_patched"):
            def _sf_load(uri, **kwargs):
                data, samplerate = sf.read(uri, dtype='float32')
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                # soundfile returns (frames, channels), torchaudio expects (channels, frames)
                tensor = torch.from_numpy(data.T)
                return tensor, samplerate
                
            torchaudio.load = _sf_load

            # Patch torchaudio.info which was completely removed in torchaudio 2.10
            class MockTorchaudioInfo:
                def __init__(self, uri):
                    info = sf.info(uri)
                    self.num_frames = info.frames
                    self.sample_rate = info.samplerate
            
            torchaudio.info = MockTorchaudioInfo
            torchaudio._patched = True

        # Get HF Token from environment (.env)
        hf_token = os.environ.get("HF_TOKEN")

        # We use a public pyannote embedding model that doesn't strictly 
        # require an HF token for this specific architecture.
        # It yields a 256-dimensional embedding.
        try:
            model = Model.from_pretrained(
                "pyannote/wespeaker-voxceleb-resnet34-LM",
                use_auth_token=hf_token
            )
            _model = Inference(model, window="whole")
            print("✅ Pyannote Voice Fingerprinting loaded.")
        except Exception as e:
            print(f"❌ Failed to load Pyannote preset. You may need an HF_TOKEN: {e}")
            raise e
        finally:
            # Restore original torch.load
            torch.load = _orig_load

    return _model

# Try to load eagerly, but don't crash the server if it fails
try:
    _get_model()
except Exception as e:
    print(f"⚠️ Pyannote model will lazy-load on first use: {e}")


def extract_voice_embedding(audio_bytes: bytes, original_filename: str, start_sec: float = None, end_sec: float = None) -> list[float]:
    """
    Given a short voice clip, extracts a robust 256-dimensional Voice Print (Embedding).
    
    1. Converts whatever audio format to 16kHz Mono WAV using FFmpeg
    2. Optional: Crops the audio using start_sec and end_sec if provided
    3. Passes it through Pyannote's WeSpeaker ResNet
    4. Returns the mathematical embedding as a float array
    """
    inference = _get_model()

    ext = os.path.splitext(original_filename)[1] or ".input"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name

    tmp_wav_path = tmp_in_path + ".wav"

    try:
        # Step 1: Force standard format (16kHz, mono) so the AI reads it correctly
        ffmpeg_cmd = ["ffmpeg", "-y"]
        if start_sec is not None and end_sec is not None:
            ffmpeg_cmd.extend(["-ss", str(start_sec), "-to", str(end_sec)])
        ffmpeg_cmd.extend(["-i", tmp_in_path, "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", tmp_wav_path])
        
        subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            check=True,
        )

        # Step 2: Extract embeddings
        # Pyannote Inference(window="whole") returns a numpy array representing the whole file
        embedding_ndarray = inference(tmp_wav_path)
        
        # Convert numpy array to flat float list
        flat_embedding = embedding_ndarray.flatten().tolist()
        
        return flat_embedding

    finally:
        for path in [tmp_in_path, tmp_wav_path]:
            try:
                os.unlink(path)
            except OSError:
                pass
