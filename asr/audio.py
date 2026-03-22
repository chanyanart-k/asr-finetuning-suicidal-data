import os
import tempfile
from pathlib import Path

import librosa
import soundfile as sf


SUPPORTED_FORMATS = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.webm']
TARGET_SR = 16_000



def prepare_audio(audio_input, output_path=None, target_sr=TARGET_SR):

    input_path = _resolve_input_path(audio_input)
    _validate_file(input_path)

    # Resample audio
    y, sr = librosa.load(str(input_path), sr=None)
    if y is None:
        raise RuntimeError("Failed to load audio file")
    
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
    # Normalized audio
    y = y / (max(abs(y)) + 1e-8)

    output_path = _write_temp_wav(y, target_sr)
    return output_path




# ------------------------------------------
#  Private helpers
# ------------------------------------------


def _resolve_input_path(audio_input) -> Path:
    """
    Write bytes / UploadedFile to a temp file and return its path.
    """
    # Handle Streamlit UploadedFile / bytes / path
    if isinstance(audio_input, (bytes, bytearray)):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f:
            f.write(audio_input)
            return Path(f.name)


    if hasattr(audio_input, "getvalue") and hasattr(audio_input, "name"):
        suffix = Path(audio_input.name).suffix or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(audio_input.getvalue())
            return Path(f.name)

    return Path(audio_input)




def _validate_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Audio File not found: {path}")
    
    if path.stat().st_size == 0:
        raise RuntimeError(f"Audio file is empty: {path}")

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {path.suffix}")
    


def _write_temp_wav(y, sr: int) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        output_path = f.name
    sf.write(output_path, y, sr)
    return output_path