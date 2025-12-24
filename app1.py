from __future__ import annotations
import queue
import threading
from app_common import build_context, console, is_exit_phrase, print_agent
from dialogue import dialogue_manager
from voice_out import VoiceOut


_speaking = threading.Event()
_tts_thread: threading.Thread | None = None
_voice = VoiceOut(rate=180, volume=1.0, voice="en-US-JennyNeural")


def is_speaking() -> bool:
    return _speaking.is_set()


def _stop_tts() -> None:
    if _speaking.is_set():
        try:
            _voice.stop()
        finally:
            _speaking.clear()


def tts_speak_async(text: str) -> None:
    cleaned = print_agent(text)
    if not cleaned:
        return

    _stop_tts()

    def _run() -> None:
        _speaking.set()
        try:
            _voice.say(cleaned)
        finally:
            _speaking.clear()

    global _tts_thread
    _tts_thread = threading.Thread(target=_run, daemon=True)
    _tts_thread.start()


def start_call() -> None:
    ctx = build_context("./docs")

    from voice_loop import VADRecorder, WhisperSTT

    stt = WhisperSTT(model_size="small")
    vad = VADRecorder(
        start_threshold=0.015,
        stop_threshold=0.020,
        silence_ms=900,
        max_utterance_s=12.0,
    )

    tts_speak_async(
        "Hello! Before we start: this conversation will be recorded for quality and claims handling in line with GDPR. "
        "By continuing, you consent to this recording. How can I help today?"
    )

    wav_queue: queue.Queue[str] = queue.Queue()
    stop_flag = threading.Event()

    def _listener() -> None:
        while not stop_flag.is_set():
            wav_path = vad.listen_utterance(is_speaking_flag=is_speaking, on_barge_in=_stop_tts)
            if wav_path:
                wav_queue.put(wav_path)

    threading.Thread(target=_listener, daemon=True).start()

    while True:
        wav_path = wav_queue.get()
        _stop_tts()

        res = stt.transcribe(wav_path)
        user_text = (res.text or "").strip()
        if not user_text:
            tts_speak_async("I didn’t catch that. Could you repeat?")
            continue

        console.print(f"[bold cyan]You[/bold cyan]: {user_text}")

        if is_exit_phrase(user_text):
            stop_flag.set()
            tts_speak_async("Goodbye!")
            break

        turn = dialogue_manager(user_text, ctx.state, ctx.rag)
        tts_speak_async(turn.response_text)

        if turn.end_call:
            stop_flag.set()
            break


if __name__ == "__main__":
    start_call()
