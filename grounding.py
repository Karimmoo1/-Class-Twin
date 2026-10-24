"""Deterministic grounding check: does the reply claim project work that is NOT in my notes?

The model likes to attach a project example to a concept answer ("in AgriVision I used a decision tree
and pruning ..."). If my notes do not say that, the sentence is invented. No LLM here, just rules:

  1. A sentence is a CLAIM if it says what I/we did ("I used", "استخدمنا", "عملت", "في مشروع" ...)
     or if it names one of my projects.
  2. Every technique term in a claim (random forest, decision tree, pruning, faiss, kubernetes ...)
     must appear in the notes of that project. If no project is named: in any note except the glossary
     (the glossary defines concepts, it is not evidence that I used them).
  3. Every number in a claim must appear in those notes.

Run   python grounding.py   for the self-test.
"""
import re
from functools import lru_cache

import config

KNOW = config.DATA_DIR / "knowledge"
GLOSSARY = "ml_glossary.md"

PROJECTS = {
    "agrivision": (r"agri ?vision|أجريفيجن|اجريفيجن",
                   ("agrivision_overview.md", "plant_disease_detection.md", "disease_rag_pipeline.md",
                    "crop_recommendation.md")),
    "face": (r"face (recognition|detection)|facenet|mtcnn", ("face_recognition_project.md",)),
    "income": (r"adult income|census|income prediction", ("adult_income_project.md",)),
    "depi": (r"\bdepi\b|class ?twin|capstone|fmcg", ("depi_genai_track_notes.md",)),
}

CLAIM = re.compile(
    r"\b(i|we)\s+(used|built|trained|applied|did|made|tried|tested|worked|implemented|chose|picked|deployed|"
    r"improved|reduced|pruned|got|reached)\b|\bin my (graduation )?(project|assignment|work)\b|\bour (project|team)\b|"
    r"استخدمت|استخدمنا|عملت|عملنا|اشتغلت|اشتغلنا|دربت|دربنا|بنيت|بنينا|طبقت|طبقنا|جربت|جربنا|قصينا|قصيت|"
    r"في مشروع|فى مشروع|مشروعي|مشروعنا", re.I)

BASE_TERMS = [
    "random forest", "decision tree", "svm", "knn", "naive bayes", "logistic regression", "mobilenet",
    "mobilenetv2", "faiss", "rag", "fine tuning", "llama", "groq", "pytorch", "tensorflow", "keras", "mtcnn",
    "facenet", "smote", "pruning", "pruned", "overfitting", "dropout", "augmentation", "transfer learning",
    "embedding", "cross validation", "validation accuracy", "training accuracy", "test accuracy", "kubernetes",
    "docker", "transformer", "bert", "gpt", "opencv", "langchain", "graphlib", "gradio", "whisper", "xtts",
    "calibration", "calibrated", "feature importance", "hyperparameter", "grid search", "early stopping",
    "bagging", "boosting", "chunking", "cosine similarity", "hallucination", "sentence transformers",
]


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[\u2010-\u2015_\-]+", " ", s)          # fine-tuning, fine‑tuning -> fine tuning
    return re.sub(r"\s+", " ", s)


def _has(text: str, term: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?:s|es)?(?![a-z0-9])", text) is not None


@lru_cache(maxsize=1)
def _notes() -> dict:
    return {p.name: _norm(p.read_text(encoding="utf-8")) for p in sorted(KNOW.glob("*"))
            if p.suffix.lower() in (".md", ".txt")}


@lru_cache(maxsize=1)
def _terms() -> list:
    terms = set(BASE_TERMS)
    g = KNOW / GLOSSARY
    if g.exists():
        for line in g.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*-\s*([^:]{2,40}):", line)
            if m:
                terms.add(_norm(m.group(1)).strip())
    return sorted(terms, key=len, reverse=True)


def clear_cache():
    _notes.cache_clear()
    _terms.cache_clear()


def _support(sentence_norm: str) -> str:
    notes = _notes()
    files = []
    for _, (pat, fs) in PROJECTS.items():
        if re.search(pat, sentence_norm):
            files += fs
    if not files:
        files = [f for f in notes if f != GLOSSARY]
    return " ".join(notes.get(f, "") for f in files)


def _numbers(s: str) -> set:
    return set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?", s))


def find_unsupported(reply_text: str) -> list:
    """-> [{"sentence": str, "unsupported": [terms and numbers not found in my notes]}]"""
    if not _notes():
        return []                                           # no notes available: cannot judge, do not block
    out = []
    for sent in re.split(r"(?<=[.!?؟؛])\s+|\n+", reply_text.strip()):
        if not sent.strip():
            continue
        sn = _norm(sent)
        project = any(re.search(p, sn) for p, _ in PROJECTS.values())
        if not (project or CLAIM.search(sent)):
            continue                                        # a plain concept sentence: nothing to check
        support = _support(sn)
        bad = [t for t in _terms() if _has(sn, t) and not _has(support, t)]
        bad = [t for t in bad if not any(t != u and t in u for u in bad)]        # keep the longest term
        known_numbers = _numbers(support)
        bad += [n for n in sorted(_numbers(sn)) if n not in known_numbers]
        if bad:
            out.append({"sentence": sent.strip(), "unsupported": bad})
    return out


def sanitize(reply_text: str):
    """Drop the unsupported sentences. -> (clean_text, removed_sentences)."""
    bad = {b["sentence"] for b in find_unsupported(reply_text)}
    keep = [s for s in re.split(r"(?<=[.!?؟؛])\s+|\n+", reply_text.strip()) if s.strip() and s.strip() not in bad]
    return " ".join(keep), sorted(bad)


# ------------------------------------------------------------------ self-test on real replies
GOOD = [  # answers the twin gave that match my notes (from the audio test runs)
    "RAG retrieves relevant text from my documents and injects it into the prompt so answers are grounded, while fine‑tuning changes the model’s weights to learn a specific task or style. In AgriVision I used a RAG pipeline with FAISS, sentence‑transformers and Llama 3.1 to give plant‑disease info with sources, because fine‑tuning would have been more work and wouldn’t provide citations.",
    "RAG بيسترجع نصوص ذات صلة من الدوكومنتات وبيحطها في الـ prompt عشان الإجابة تكون grounded، أما fine‑tuning فبيعدل وزن الـ LLM عشان يتعلم مهمة أو ستايل معين. في مشروع AgriVision استخدمت RAG pipeline مع FAISS وSentence‑Transformers وLlama 3.1 عشان أدي معلومات عن أمراض النبات مع المصادر، لأن fine‑tuning كان هيكون شغل أكتر ومش هيدّي citations.",
    "في AgriVision استخدمت Random Forest متظبط (calibrated) عشان أرشح المحصول من 22 نوع، وبيدّي feature importance.",
    "الـ pruning في الـ decision tree هو قص الفروع اللي مالهاش فرصة تكسب أو اللي بتسبب overfitting عشان الشجرة تبقى أبسط.",
    "Overfitting means the model memorizes the training data: high training accuracy but low validation accuracy.",
    "I haven't worked on that, so I don't want to guess. I can look into it and get back to you.",
]
BAD = [   # invented claims the twin really produced, plus two synthetic ones
    "البروننج هو قص الفروع اللي مالهاش فائدة. في مشروع AgriVision استخدمت decision tree لتوصية المحاصيل وعملت pruning عشان الشجرة ما تحفظش noise وتدي توصيات أكتر ثابتة.",
    "في مشروع الـ DEPI استخدمنا decision tree لتحديد سياسات الرد للـ agent وقصينا الفروع اللي ماكانتش بتتفاعل مع أي طلب، وده حسّن الـ validation accuracy وقلل الـ overfitting.",
    "I used Kubernetes in my assignment to deploy the model.",
    "In AgriVision we reached 98% accuracy with MobileNetV2.",
]


if __name__ == "__main__":
    ok = True
    for t in GOOD:
        r = find_unsupported(t)
        ok &= not r
        print("OK   " if not r else "WRONGLY FLAGGED", t[:70].replace("\n", " "), r)
    for t in BAD:
        r = find_unsupported(t)
        ok &= bool(r)
        print("CAUGHT" if r else "MISSED ", t[:70], [(x["unsupported"]) for x in r])
    print("\nALL GOOD" if ok else "\nPROBLEM")
