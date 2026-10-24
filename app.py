"""Gradio demo. Run:  python app.py   (in Colab: !python app.py or import and call demo.launch(share=True))"""
import gradio as gr

from retrieval import warmup
from twin import MEMORY, run_turn


def handle(audio, typed, muted):
    if not audio and not (typed or "").strip():
        return "Record, upload, or type a question first.", "", [], {}, {}, None, {}
    out = run_turn(audio_path=audio, text=(typed or "").strip() or None, muted=muted)
    tr = out["transcript"]
    shown = f"{tr['text']}\n\n[language: {tr['language']} | whisper: {tr['whisper_language']} | quality: {tr['quality']}]"
    return shown, out["tree"], out["tool_calls"], out["reply"], out["plan"], out["audio"], out["timings"]


def reset():
    MEMORY.reset()
    return "Memory cleared."


with gr.Blocks(title="Class Twin") as demo:
    gr.Markdown("## Class Twin - Phase 1\nRun only in meetings where the instructor knows the twin is on. "
                "Never for attendance, quizzes or exams.")
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Instructor question")
            typed = gr.Textbox(label="...or type it (skips speech-to-text, for debugging)")
            muted = gr.Checkbox(label="Kill switch: mute the twin (no audio)")
            go = gr.Button("Run", variant="primary")
            clear = gr.Button("Clear meeting memory")
            status = gr.Textbox(label="Status", interactive=False)
        with gr.Column():
            transcript = gr.Textbox(label="Transcript", lines=3)
            tree = gr.Textbox(label="Decision tree (with pruned branches)", lines=14)
    with gr.Row():
        calls = gr.JSON(label="Tool calls + JSON arguments")
        final = gr.JSON(label="Final TwinReply")
    with gr.Row():
        plan = gr.JSON(label="Dependency graph run (waves = parallel groups)")
        timings = gr.JSON(label="Stage timings (seconds)")
    out_audio = gr.Audio(label="Twin voice", type="filepath")
    go.click(handle, [audio, typed, muted], [transcript, tree, calls, final, plan, out_audio, timings])
    clear.click(reset, None, status)

warmup()   # build the search indexes once, before the first question

if __name__ == "__main__":
    demo.launch(share=True)
