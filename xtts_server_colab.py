"""Run this on Colab (GPU runtime) to host XTTS-v2 with YOUR voice. Your laptop calls it with gradio_client.

Colab cells:
    !pip install -q coqui-tts gradio soundfile
    # (Colab already has torch. If pip complains about torch, install the torch version it asks for.)
    import os
    os.environ["TWIN_TTS_PASS"] = "choose-a-password"          # the share link is public: keep the password
    os.environ["TWIN_VOICE_DIR"] = "/content/drive/MyDrive/ClassTwin/twin_data/voice"
    %run xtts_server_colab.py

Copy the public URL it prints into the laptop:  XTTS_URL=<url>  TWIN_TTS_PASS=<password>
Your voice samples (english_sample.wav, arabic_sample.wav) never leave your Drive + this Colab session.
Only use YOUR OWN voice, with your consent recording made.
"""
import os
import tempfile

import numpy as np
import soundfile as sf
import torch

os.environ.setdefault("COQUI_TOS_AGREED", "1")          # XTTS-v2 license: non-commercial (CPML)
from TTS.api import TTS                                 # noqa: E402
import gradio as gr                                     # noqa: E402

VOICE_DIR = os.environ.get("TWIN_VOICE_DIR", "/content/drive/MyDrive/ClassTwin/twin_data/voice")
SAMPLES = {"en": ["english_sample.wav"], "ar": ["arabic_sample.wav"]}
PASSWORD = os.environ.get("TWIN_TTS_PASS", "")
USER = "twin"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
SR = int(tts.synthesizer.output_sample_rate)

wavs = {k: [os.path.join(VOICE_DIR, f) for f in v] for k, v in SAMPLES.items()}
for k, v in wavs.items():
    for p in v:
        assert os.path.exists(p), f"missing voice sample: {p}"

# Speed-up: compute the voice conditioning once per language, not on every sentence.
model = tts.synthesizer.tts_model
LATENTS = {}
try:
    for k, v in wavs.items():
        LATENTS[k] = model.get_conditioning_latents(audio_path=v)
    print("fast path: conditioning latents cached")
except Exception as e:
    print("fast path not available, using tts.tts() instead:", e)


def synthesize(text: str, lang: str):
    lang = "en" if lang == "en" else "ar"
    if lang in LATENTS:
        out = model.inference(text, lang, LATENTS[lang][0], LATENTS[lang][1], temperature=0.65)
        wav = np.asarray(out["wav"], dtype=np.float32)
    else:
        wav = np.asarray(tts.tts(text=text, speaker_wav=wavs[lang], language=lang), dtype=np.float32)
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(path, wav, SR)
    return path


# warm up so the first real sentence is fast
synthesize("Hello.", "en")
synthesize("مرحبا.", "ar")

demo = gr.Interface(fn=synthesize,
                    inputs=[gr.Textbox(label="text"), gr.Dropdown(["en", "ar"], value="en", label="lang")],
                    outputs=gr.Audio(type="filepath"), api_name="synthesize")
demo.queue(max_size=8).launch(share=True, auth=(USER, PASSWORD) if PASSWORD else None, debug=False)
