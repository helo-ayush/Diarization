"""
patch_pyannote.py
-----------------
Run this ONCE after `pip install -r requirements.txt` to fix
Pyannote 3.1.1 incompatibilities with modern Python packages.

WHY IS THIS NEEDED? (The 3 Major Ecosystem Shifts)
--------------------------------------------------
1. HuggingFace Hub Issue (The `is_offline_mode` error):
   pyannote.audio (v3.1.1) hardcoded the use of `use_auth_token` for authentication.
   HuggingFace recently removed `use_auth_token` completely (replacing it with `token`).
   Because of this, Pyannote forcefully downgraded `huggingface_hub` to an ancient version, 
   breaking modern `transformers` installations. This script patches the pyannote source 
   code to natively use `token` instead of `use_auth_token`.

2. PyTorch 2.6 Security Issue (The `WeightsUnpickler` error):
   PyTorch 2.6 changed the default behavior of `torch.load()` to strictly block the loading 
   of serialized Python classes (`weights_only=True`) for security reasons. Pyannote relies 
   heavily on pickled model configurations. This is handled via a dynamic monkey-patch in
   `fingerprint_processor.py` that temporarily sets `weights_only=False` when loading models.

3. TorchAudio 2.10 Windows Issue (The `TorchCodec` / `FFmpeg` error):
   TorchAudio 2.10 ripped out their traditional audio-loading backends and replaced them 
   with `torchcodec`. On Windows, `torchcodec` fails entirely without complex system-level 
   FFmpeg C++ DLLs. This is also handled in `fingerprint_processor.py` by intercepting 
   TorchAudio requests and routing them to `soundfile` (pure Python) instead!

Additional Fixes (applied below):
  1. pyannote/audio/core/io.py        — torchaudio.set_audio_backend() removed in torchaudio 2.x
  2. pyannote/audio/core/inference.py  — np.NaN removed in NumPy 2.0
  3. pyannote/audio/tasks/.../mixins.py — np.NaN removed in NumPy 2.0
  4. pyannote/audio/tasks/.../speaker_diarization.py — np.NaN removed in NumPy 2.0

Usage:  python patch_pyannote.py
"""
import site
import os
import sys

def find_file(relative_path: str) -> str | None:
    for sp in sys.path:
        full_path = os.path.join(sp, relative_path)
        if os.path.exists(full_path):
            return full_path
    return None

def patch_file(rel_path: str, old: str, new: str, description: str) -> bool:
    path = find_file(rel_path)
    if not path:
        print(f"  ⚠️  SKIPPED — file not found: {rel_path}")
        return False

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if old not in content:
        print(f"  ✅ Already patched: {description}")
        return True

    content = content.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  ✅ Patched: {description}")
    return True


print("\n🩹 Applying Pyannote compatibility patches for Python 3.13 / torchaudio 2.x / NumPy 2.x...\n")

patches = [
    (
        "pyannote/audio/core/io.py",
        'torchaudio.set_audio_backend("soundfile")',
        '# torchaudio.set_audio_backend removed in torchaudio 2.x — patched for compatibility',
        "io.py — remove deprecated torchaudio.set_audio_backend()",
    ),
    (
        "pyannote/audio/core/inference.py",
        "missing: float = np.NaN,",
        "missing: float = np.nan,",
        "inference.py — np.NaN -> np.nan",
    ),
    (
        "pyannote/audio/tasks/segmentation/mixins.py",
        "y[y == 0] = np.NaN",
        "y[y == 0] = np.nan",
        "mixins.py — np.NaN -> np.nan",
    ),
    (
        "pyannote/audio/tasks/segmentation/speaker_diarization.py",
        "y[y == 0] = np.NaN",
        "y[y == 0] = np.nan",
        "speaker_diarization.py — np.NaN -> np.nan",
    ),
    (
        "pyannote/audio/core/model.py",
        "use_auth_token=use_auth_token,",
        "token=use_auth_token,",
        "model.py — hf_hub_download use_auth_token -> token",
    ),
    (
        "pyannote/audio/core/pipeline.py",
        "use_auth_token=use_auth_token,",
        "token=use_auth_token,",
        "pipeline.py — hf_hub_download use_auth_token -> token",
    ),
]

success = True
for args in patches:
    if not patch_file(*args):
        success = False

if success:
    print("\n✅  All patches applied! Pyannote is ready to use.\n")
else:
    print("\n❌  Some patches failed. Is pyannote.audio installed? Run: pip install pyannote.audio==3.1.1\n")
    sys.exit(1)
