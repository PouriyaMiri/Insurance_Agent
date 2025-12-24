from __future__ import annotations
import asyncio
import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Optional
import sounddevice as sd
import soundfile as sf


@dataclass(frozen=True)
class EdgeTTSConfig:
    voice: str = "en-US-JennyNeural"
    rate: str = "+0%"
    volume: str = "+0%"


class VoiceOut:
    def __init__(
        self,
        rate: int = 180,
        volume: float = 1.0,
        voice: Optional[str] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._stream: Optional[sd.OutputStream] = None

        self.rate = max(80, int(rate))
        self.volume = max(0.0, float(volume))
        self.voice = voice or "en-US-JennyNeural"

        self._cfg = EdgeTTSConfig(
            voice=self.voice,
            rate=self._format_edge_rate(self.rate),
            volume=self._format_edge_volume(self.volume),
        )

    @staticmethod
    def _format_edge_rate(rate_wpm: int) -> str:
        try:
            perc = (rate_wpm / 180.0 - 1.0) * 100.0
            return f"{perc:+.0f}%"
        except Exception:
            return "+0%"

    @staticmethod
    def _format_edge_volume(volume: float) -> str:
        try:
            vol_perc = (volume - 1.0) * 100.0
            return f"{vol_perc:+.0f}%"
        except Exception:
            return "+0%"

    def stop(self) -> None:
        self._stop_event.set()
        try:
            if self._stream is not None:
                self._stream.stop()
        except Exception:
            pass
        try:
            sd.stop()
        except Exception:
            pass

    async def _edge_speech_to_file(self, text: str, outfile: str) -> None:
        import edge_tts  # type: ignore

        communicate = edge_tts.Communicate(
            text=text,
            voice=self._cfg.voice,
            rate=self._cfg.rate,
            volume=self._cfg.volume,
        )
        await communicate.save(outfile)

    def _play_audio_file(self, path: str) -> None:
        data, samplerate = sf.read(path, dtype="float32")
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self._stop_event.clear()
        pos = 0

        def callback(outdata, frames, time_info, status) -> None:  # noqa: ARG001
            nonlocal pos

            if self._stop_event.is_set():
                outdata.fill(0)
                raise sd.CallbackStop()

            chunk = data[pos: pos + frames]
            if len(chunk) < frames:
                outdata[: len(chunk)] = chunk
                outdata[len(chunk):] = 0
                raise sd.CallbackStop()

            outdata[:] = chunk
            pos += frames

        try:
            with sd.OutputStream(
                samplerate=samplerate,
                channels=int(data.shape[1]),
                dtype="float32",
                callback=callback,
            ) as stream:
                self._stream = stream
                while stream.active:
                    sd.sleep(20)
        finally:
            self._stream = None
            try:
                os.remove(path)
            except Exception:
                pass

    def say(self, text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return

        with self._lock:
            self._stop_event.clear()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            outfile = tmp.name
            tmp.close()

            try:
                asyncio.run(self._edge_speech_to_file(cleaned, outfile))
                self._play_audio_file(outfile)
            except Exception:
                if os.getenv("VOICEOUT_ECHO_ON_FAIL") == "1":
                    print(cleaned)
                try:
                    os.remove(outfile)
                except Exception:
                    pass
