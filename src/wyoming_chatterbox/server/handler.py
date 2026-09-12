"""Wyoming event handler for Chatterbox TTS."""

from __future__ import annotations

import asyncio
import logging

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import (
    Attribution,
    Describe,
    Info,
    SelectProgram,
    TtsProgram,
    TtsVoice,
)
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from wyoming_chatterbox.config import Settings
from wyoming_chatterbox.models.base import ChatterboxBackend
from wyoming_chatterbox.segmentation.segmenter import TextSegmenter
from wyoming_chatterbox.synthesis.pipeline import SynthesisPipeline
from wyoming_chatterbox.voices.manager import VoiceManager

logger = logging.getLogger(__name__)

_ATTRIBUTION = Attribution(
    name="Resemble AI",
    url="https://github.com/resemble-ai/chatterbox",
)


class ChatterboxEventHandler(AsyncEventHandler):
    """Handle Wyoming protocol events for one client connection."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        backends: dict[str, ChatterboxBackend],
        settings: Settings,
        voice_manager: VoiceManager,
        default_variant: str,
    ) -> None:
        super().__init__(reader, writer)

        self._backends = backends
        self._settings = settings
        self._voice_manager = voice_manager
        self._active_variant = default_variant

        self._pipeline = SynthesisPipeline(
            backends[default_variant],
            settings,
            voice_manager,
        )

        # Streaming-input state.
        self._streaming = False
        self._stream_segmenter: TextSegmenter | None = None
        self._stream_parts: list[str] = []
        self._stream_queue: asyncio.Queue[str | None] | None = None
        self._stream_task: asyncio.Task[None] | None = None
        self._stream_voice: str | None = None
        self._stream_language = settings.chatterbox_default_language

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self._handle_describe()
            return True

        if SelectProgram.is_type(event.type):
            await self._handle_select_program(SelectProgram.from_event(event))
            return True

        if SynthesizeStart.is_type(event.type):
            await self._handle_synthesize_start(SynthesizeStart.from_event(event))
            return True

        if SynthesizeChunk.is_type(event.type):
            await self._handle_synthesize_chunk(SynthesizeChunk.from_event(event))
            return True

        if SynthesizeStop.is_type(event.type):
            await self._handle_synthesize_stop()
            return True

        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)

            # Streaming Wyoming clients send the complete Synthesize
            # message as a backwards-compatibility event. The chunks are
            # authoritative, so don't synthesize the response twice.
            if self._streaming:
                logger.debug("Ignoring compatibility Synthesize during active stream")
                return True

            await self._handle_synthesize(synthesize)
            return True

        return True

    # -- describe ---------------------------------------------------------

    async def _handle_describe(self) -> None:
        await self.write_event(self._build_info().event())

    def _build_info(self) -> Info:
        voice_names = self._voice_manager.list_voices()
        programs: list[TtsProgram] = []

        for variant, backend in self._backends.items():
            languages = backend.supported_languages()

            # Always advertise the model's built-in/default voice.
            #
            # Home Assistant derives a Wyoming TTS provider's supported
            # languages from its installed voices. Without at least one
            # advertised voice, the provider appears in HA but cannot be
            # selected in an Assist pipeline.
            voices = [
                TtsVoice(
                    name="default",
                    description="Chatterbox default voice",
                    attribution=_ATTRIBUTION,
                    installed=True,
                    version=None,
                    languages=list(languages),
                )
            ]

            voices.extend(
                TtsVoice(
                    name=name,
                    description=f"Reference voice {name}",
                    attribution=_ATTRIBUTION,
                    installed=True,
                    version=None,
                    languages=list(languages),
                )
                for name in voice_names
                if name != "default"
            )

            programs.append(
                TtsProgram(
                    name=variant,
                    description=f"Chatterbox TTS ({variant})",
                    attribution=_ATTRIBUTION,
                    installed=backend.is_loaded,
                    version=None,
                    voices=voices,
                    supports_synthesize_streaming=True,
                )
            )

        return Info(tts=programs)

    # -- select program ---------------------------------------------------

    async def _handle_select_program(
        self,
        event: SelectProgram,
    ) -> None:
        if self._streaming:
            await self.write_event(
                Error(text="Cannot change TTS program during active synthesis").event()
            )
            return

        if event.name not in self._backends:
            await self.write_event(Error(text=f"Unknown program: {event.name}").event())
            return

        self._active_variant = event.name

        self._pipeline.close()
        self._pipeline = SynthesisPipeline(
            self._backends[event.name],
            self._settings,
            self._voice_manager,
        )

        logger.debug(
            "Selected program %s",
            event.name,
        )

    # -- shared request helpers ------------------------------------------

    def _resolve_voice_language(
        self,
        voice_event,
    ) -> tuple[str | None, str]:
        voice = None
        language = self._settings.chatterbox_default_language

        if voice_event is not None:
            voice = voice_event.name or None

            if voice_event.language:
                language = voice_event.language

        if not voice and self._settings.chatterbox_default_voice:
            voice = self._settings.chatterbox_default_voice

        return voice, language

    async def _emit_audio_chunk(
        self,
        chunk: bytes,
        sample_rate: int,
    ) -> None:
        if not chunk:
            return

        await self.write_event(
            AudioChunk(
                audio=chunk,
                rate=sample_rate,
                width=2,
                channels=1,
            ).event()
        )

    # -- complete-text synthesis -----------------------------------------

    async def _handle_synthesize(
        self,
        event: Synthesize,
    ) -> None:
        try:
            backend = self._backends[self._active_variant]

            if not backend.is_loaded:
                backend.load()

            voice, language = self._resolve_voice_language(event.voice)

            sample_rate = backend.sample_rate

            await self.write_event(
                AudioStart(
                    rate=sample_rate,
                    width=2,
                    channels=1,
                ).event()
            )

            async for chunk in self._pipeline.synthesize_stream(
                event.text,
                voice=voice,
                language=language,
            ):
                await self._emit_audio_chunk(
                    chunk,
                    sample_rate,
                )

            await self.write_event(AudioStop().event())

        except Exception as exc:  # noqa: BLE001
            logger.exception("Synthesis error")

            await self.write_event(Error(text=str(exc)).event())

    # -- streaming-text synthesis ----------------------------------------

    async def _handle_synthesize_start(
        self,
        event: SynthesizeStart,
    ) -> None:
        if self._streaming:
            await self.write_event(
                Error(text="A streaming synthesis request is already active").event()
            )
            return

        backend = self._backends[self._active_variant]

        if not backend.is_loaded:
            backend.load()

        voice, language = self._resolve_voice_language(event.voice)

        self._streaming = True
        self._stream_voice = voice
        self._stream_language = language
        self._stream_parts = []
        self._stream_queue = asyncio.Queue()

        # Segmented mode begins synthesis as soon as a complete phrase
        # becomes available. Buffered/off modes retain the full text.
        if self._settings.chatterbox_streaming_mode == "segmented":
            self._stream_segmenter = TextSegmenter(
                min_chars=self._settings.chatterbox_segment_min_chars,
                target_chars=self._settings.chatterbox_segment_target_chars,
                max_chars=self._settings.chatterbox_segment_max_chars,
            )
        else:
            self._stream_segmenter = None

        # Send the PCM format immediately. Actual audio follows as soon
        # as the first phrase has been synthesized.
        await self.write_event(
            AudioStart(
                rate=backend.sample_rate,
                width=2,
                channels=1,
            ).event()
        )

        self._stream_task = asyncio.create_task(
            self._stream_audio_worker(),
            name="chatterbox-streaming-synthesis",
        )

        logger.debug(
            "Streaming synthesis started: variant=%s voice=%s language=%s mode=%s",
            self._active_variant,
            voice,
            language,
            self._settings.chatterbox_streaming_mode,
        )

    async def _handle_synthesize_chunk(
        self,
        event: SynthesizeChunk,
    ) -> None:
        if not self._streaming or self._stream_queue is None:
            await self.write_event(
                Error(text=("Received SynthesizeChunk without SynthesizeStart")).event()
            )
            return

        if self._stream_segmenter is None:
            self._stream_parts.append(event.text)
            return

        for segment in self._stream_segmenter.feed(event.text):
            logger.debug(
                "Streaming segment ready: %r",
                segment,
            )

            await self._stream_queue.put(segment)

    async def _handle_synthesize_stop(self) -> None:
        if not self._streaming or self._stream_queue is None:
            await self.write_event(
                Error(text=("Received SynthesizeStop without active stream")).event()
            )
            return

        try:
            if self._stream_segmenter is not None:
                for segment in self._stream_segmenter.flush():
                    await self._stream_queue.put(segment)

            else:
                text = "".join(self._stream_parts).strip()

                if text:
                    await self._stream_queue.put(text)

            # End-of-input sentinel.
            await self._stream_queue.put(None)

            if self._stream_task is not None:
                await self._stream_task

        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming synthesis error")

            await self.write_event(Error(text=str(exc)).event())

        finally:
            await self.write_event(AudioStop().event())

            await self.write_event(SynthesizeStopped().event())

            self._reset_stream_state()

    async def _stream_audio_worker(self) -> None:
        """Synthesize queued phrases while more text arrives."""

        assert self._stream_queue is not None

        backend = self._backends[self._active_variant]
        sample_rate = backend.sample_rate

        while True:
            segment = await self._stream_queue.get()

            if segment is None:
                return

            logger.debug(
                "Synthesizing streaming segment: %r",
                segment,
            )

            async for chunk in self._pipeline.synthesize_stream(
                segment,
                voice=self._stream_voice,
                language=self._stream_language,
            ):
                await self._emit_audio_chunk(
                    chunk,
                    sample_rate,
                )

    def _reset_stream_state(self) -> None:
        self._streaming = False
        self._stream_segmenter = None
        self._stream_parts = []
        self._stream_queue = None
        self._stream_task = None
        self._stream_voice = None
        self._stream_language = self._settings.chatterbox_default_language
