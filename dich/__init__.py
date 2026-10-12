"""Dịch — Vietnamese → English translator running a local MLX model.

Run the app with `python -m dich`. Submodules:

  ui            Tkinter app (Translate / Speech / Settings tabs)
  backend       extraction, prompts, cache, engine, contextual translation
  speech_engine mic + file audio, Whisper Vietnamese → completed sentences
  theme         palette and shared styles
"""
