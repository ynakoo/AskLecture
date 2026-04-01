"""
Whisper transcription module.
Uses locally-installed OpenAI Whisper model to transcribe audio files.
"""

import whisper

# Module-level cache for the loaded model
_whisper_model = None
_loaded_model_size = None


def load_whisper_model(model_size: str = "base") -> whisper.Whisper:
    """
    Loads the Whisper model, caching it after the first load.

    Args:
        model_size: Size of the Whisper model to load.
                    Options: "tiny", "base", "small", "medium", "large".

    Returns:
        Loaded Whisper model instance.
    """
    global _whisper_model, _loaded_model_size

    if _whisper_model is not None and _loaded_model_size == model_size:
        return _whisper_model

    print(f"[Whisper] Loading '{model_size}' model (first time may download ~140MB)...")
    _whisper_model = whisper.load_model(model_size)
    _loaded_model_size = model_size
    print(f"[Whisper] Model '{model_size}' loaded successfully.")

    return _whisper_model
