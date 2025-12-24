from __future__ import annotations
import queue
import tempfile
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np
import sounddevice as sd
import soundfile as sf

@dataclass
class STTResult:
    text: str
    language: Optional[str] = None

class VADRecorder:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        frame_ms: int = 30,
        start_threshold: float = 0.015,   # tune for your mic
        stop_threshold: float = 0.010,    # tune for your mic
        silence_ms: int = 700,
        max_utterance_s: float = 12.0,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.start_threshold = start_threshold
        self.stop_threshold = stop_threshold
        self.silence_frames = int(silence_ms / frame_ms)
        self.max_frames = int(max_utterance_s * 1000 / frame_ms)

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(x))))

    def listen_utterance(self, is_speaking_flag, on_barge_in=None) -> Optional[str]:
        q_audio: "queue.Queue[np.ndarray]" = queue.Queue()

        def callback(indata, frames, time_info, status):
            q_audio.put(indata.copy())

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.frame_samples,
            callback=callback,
        ):
            started = False
            buffer = []
            silent_count = 0
            frames_count = 0

            while True:
                frame = q_audio.get()
                mono = frame[:, 0] if frame.ndim > 1 else frame
                e = self._rms(mono)

                if not started:
                    if e >= self.start_threshold:
                        if is_speaking_flag() and callable(on_barge_in):
                            try:
                                on_barge_in()
                            except Exception:
                                pass
                        started = True
                        buffer.append(frame)
                        frames_count = 1
                        silent_count = 0
                    else:
                        continue
                else:
                    buffer.append(frame)
                    frames_count += 1

                    if e < self.stop_threshold:
                        silent_count += 1
                    else:
                        silent_count = 0

                    if silent_count >= self.silence_frames or frames_count >= self.max_frames:
                        break

            if not buffer:
                return None

            audio = np.concatenate(buffer, axis=0)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(tmp.name, audio, self.sample_rate)
            return tmp.name

class WhisperSTT:
    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, wav_path: str) -> STTResult:
        segments, info = self.model.transcribe(wav_path, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return STTResult(text=text, language=getattr(info, "language", None))
