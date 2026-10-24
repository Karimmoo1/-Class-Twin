# Class Twin

An AI twin that joins an online class, listens, and answers questions **in my own voice**, in **Egyptian Arabic, English, or a mix of both**. Built as a two-phase project in the DEPI *Agentic AI & Generative AI* track.

> **Responsible use.** Only my own voice, with my consent recording. Only in meetings where everyone knows the twin is running (its display name ends with "(AI twin)"). Never for attendance, quizzes or exams. A kill switch stops it within one second. Voice samples and clones are not in this repo.

## What it does
```
meeting audio -> VAD (silero) -> speech-to-text (Groq Whisper) -> decision tree (answer / defer / stay silent / ask to repeat)
              -> answer loop (LLM tool calling over my notes, with a deterministic goal-check) -> cloned voice (XTTS-v2) -> meeting
```
- **Decision tree** with pruning: cheap rule checks first, the LLM only for what is left.
- **Answer loop**: the model picks tools (`search_my_notes`, `get_style_examples`, `lookup_glossary`, `get_profile`); tool errors are observations; identical failed calls are blocked; max 4 iterations, then a safe fallback.
- **Goal-check before speaking**: not empty, right language, right length, cites a source, and **claims are grounded in my notes** (`grounding.py` blocks invented project experience).
- **Conversation**: greetings in Egyptian Arabic / English, follow-ups without my name, "repeat" / "I didn't understand" / "continue" requests.
- **Human in the loop**: approve or auto mode, kill switch, live state on screen. One JSONL log record per utterance.

## Repo layout
```
class-twin/
├─ README.md  LICENSE  requirements.txt  .gitignore
│
├─ config.py  schemas.py  textutils.py  memory.py  retrieval.py        Phase 1: settings, language tools, memory, FAISS search
├─ tools.py   stt.py  llm.py  decide.py  planner.py  speak.py  twin.py   Phase 1: tools, STT, decision tree, reply graph, one full turn
├─ app.py  run_tests.py                                                  Phase 1: Gradio demo + the 8 tests
│
├─ live_config.py  agent_loop.py  answer_loop.py  grounding.py           Phase 2: the loop, the answer loop, the grounding check
├─ context_mgr.py  vad.py  audio_io.py  controls.py  speaker.py          Phase 2: memory + summary, VAD, audio in/out, keys, speaking
├─ tts_xtts.py  tashkeel.py  live_answer.py                              Phase 2: cloned voice, harakat for the voice, live policy (greetings, conversation)
├─ meeting_loop.py  livelog.py  run_live.py                              Phase 2: Observe-Think-Act loop, JSONL log, entry point
├─ xtts_server_colab.py  tashkeel_ab_test.py  latency_report.py          Phase 2: Colab voice server, harakat A/B test, latency table
├─ run_forced_tests.py  test_live_layer.py  test_tashkeel.py             Phase 2: the 5 forced tests + extra tests
│
├─ twin_data/                      my notes (safe to publish)
│  ├─ profile.json  style_examples.jsonl
│  └─ knowledge/*.md               project notes the twin may answer from
├─ tests/tests.json                the 8 test questions (audio files are NOT in the repo)
├─ docs/                           README_phase1.md, README_phase2.md, screenshots, demo link
└─ outputs/live/                   forced_tests_output.txt, latency_table.md, sample_log.jsonl (cleaned)
```

## Quick start
```bash
pip install -r requirements.txt
export GROQ_API_KEY=...                  # Windows PowerShell: $env:GROQ_API_KEY="..."
python run_tests.py --mode text          # Phase 1: 8 tests on typed questions
python run_forced_tests.py               # Phase 2: 5 forced tests, offline (no key, no audio)
python test_live_layer.py                # greetings / conversation / Egyptian Arabic tests, offline
python run_live.py --dry --source mic    # talk to it, it prints what it would say
```
Live meeting (Windows + VB-CABLE, voice server on Colab): see [`docs/README_phase2.md`](docs/README_phase2.md).

## Results
| | |
|---|---|
| Phase 1 tests (audio, 8 cases) | **8 / 8** actions as expected |
| Forced tests (Phase 2) | **5 / 5** + extra checks, see `outputs/live/forced_tests_output.txt` |
| Latency (median / worst per stage) | see `outputs/live/latency_table.md` |
| Voice blind test | **__ / 6** identified correctly |

## What went wrong, and what I changed (short version)
- The model invented project experience (e.g. "I used pruning in AgriVision"). Prompts only reduced it, so I added a **deterministic grounding check** that blocks claims not found in my notes.
- Groq's free tier (8,000 tokens/minute, 200,000 tokens/day) ran out during the first live test.
- Whisper writes Egyptian-Arabic technical words badly (for example "MobileNetV2" or "fine-tuning" inside Arabic sentences).
- Replies were too long to say in a class (30 to 50 seconds of speech), and the cloned voice takes several seconds for its first sentence.

## Known limits
XTTS-v2's Arabic is standard-Arabic based, so the accent is not fully Egyptian. Free-tier Groq limits can add waiting time. Whisper mis-hears names and mixed Arabic/English terms.

## Credits and licenses
Code: MIT. Models keep their own licenses (XTTS-v2 is **non-commercial**). Built by Karim Mohamed Saeed Tohamy, Computer Science (AI), Banha University.
