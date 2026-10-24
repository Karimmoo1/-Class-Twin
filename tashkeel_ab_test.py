"""Listen and decide: does the voice sound better WITH harakat?
Needs the XTTS server running (same settings as run_live):  XTTS_URL and TWIN_TTS_PASS set.
    python tashkeel_ab_test.py
Writes pairs to outputs/live/ab/:  1_plain.wav / 1_harakat.wav, 2_plain.wav / 2_harakat.wav ...
Play each pair. If the harakat version is clearly better and still sounds Egyptian-enough: use  --tashkeel llm."""
import soundfile as sf

import live_config as lc
import tashkeel
from tts_xtts import XTTSVoice

SENTENCES = [
    "أهلاً بيك! أنا التوأم الذكي بتاع كريم. أقدر أساعدك بإيه؟",
    "الـ RAG بيجيب الـ notes المناسبة الأول وبيحطها في الـ prompt عشان الإجابة تبقى مبنية على مصادر.",
    "الـ overfitting معناه إن الموديل بيحفظ الداتا بدل ما يفهمها، فبيطلع نتيجة حلوة في التدريب وضعيفة في الاختبار.",
    "خليني أراجعها وأرد على حضرتك.",
]
tashkeel.MODE = "llm"
voice = XTTSVoice(mode=lc.XTTS_MODE, url=lc.XTTS_URL)
out = lc.LIVE_DIR / "ab"
out.mkdir(exist_ok=True)
for i, text in enumerate(SENTENCES, 1):
    marked = tashkeel.diacritize(text)
    print(f"\n{i}. plain  : {text}\n   harakat: {marked}")
    if marked == text:
        print("   (harakat were NOT added: the check rejected the model's answer or the LLM failed)")
    for name, t in (("plain", text), ("harakat", marked)):
        audio, sr = voice.synth(t, "ar")
        sf.write(out / f"{i}_{name}.wav", audio, sr)
print("\nsaved to", out)
