#!/usr/bin/env python3
"""
Benchmark local chatbot latency without using the microphone.

Examples:
    python -m scripts.benchmark_latency --llm
    python -m scripts.benchmark_latency --tts
    python -m scripts.benchmark_latency --stt
    python -m scripts.benchmark_latency --all
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.models.tts import TextToSpeech


LLM_PROMPTS = [
    "Hey, what's up?",
    "I had a really rough day at work.",
    "What's a quick way to make dinner with eggs and rice?",
    "I just got promoted!",
]

TTS_TEXTS = [
    "Yeah, that sounds really frustrating.",
    "That's huge. You must be so excited.",
    "Try frying the rice first, then stir in the eggs at the end.",
    "I can help with that, but I need one more detail.",
]

# Fixture sentences for the STT benchmark. Keywords are checked against the
# transcript so accuracy regressions show up alongside latency numbers.
STT_FIXTURES = [
    ("fox", "The quick brown fox jumps over the lazy dog.",
     ["quick", "brown", "fox", "lazy", "dog"]),
    ("meeting", "Please schedule a meeting for Tuesday at three thirty.",
     ["schedule", "meeting", "tuesday"]),
    ("pizza", "I would like to order a large pepperoni pizza with extra cheese.",
     ["order", "pepperoni", "pizza", "cheese"]),
    ("weather", "The weather in San Francisco is foggy in the morning.",
     ["weather", "francisco", "foggy", "morning"]),
]

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p90": 0.0}
    ordered = sorted(values)
    p90_idx = min(len(ordered) - 1, int((len(ordered) - 1) * 0.9))
    return {
        "avg": statistics.fmean(values),
        "p50": statistics.median(values),
        "p90": ordered[p90_idx],
    }


def clear_mlx_cache() -> None:
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def benchmark_llm(model_names: list[str], max_tokens: int) -> list[dict[str, Any]]:
    from src.models.llm import LanguageModel

    results = []
    for model_name in model_names:
        llm = LanguageModel(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=settings.llm.temperature,
            top_p=settings.llm.top_p,
            top_k=settings.llm.top_k,
            min_p=settings.llm.min_p,
            history_turns=0,
            enable_thinking=settings.llm.enable_thinking,
            system_prompt=settings.llm.system_prompt,
        )
        llm.warmup()

        model_results = []
        for prompt in LLM_PROMPTS:
            llm.clear_history()
            start = time.time()
            response = "".join(llm.generate_stream(prompt))
            total = time.time() - start
            stats = dict(llm.last_stream_stats)
            model_results.append({
                "prompt": prompt,
                "response": llm.clean_response_text(response),
                "total": total,
                "first_token": stats.get("first_token", 0.0),
                "generation_tps": stats.get("generation_tps", 0.0),
                "prompt_tokens": stats.get("prompt_tokens", 0),
                "prompt_tps": stats.get("prompt_tps", 0.0),
                "words": len(llm.clean_response_text(response).split()),
            })

        results.append({
            "model": model_name,
            "summary": {
                "first_token": summarize([r["first_token"] for r in model_results]),
                "total": summarize([r["total"] for r in model_results]),
                "generation_tps": summarize([r["generation_tps"] for r in model_results]),
                "words": summarize([float(r["words"]) for r in model_results]),
            },
            "runs": model_results,
        })

        del llm
        gc.collect()
        clear_mlx_cache()

    return results


def benchmark_tts(voice: str) -> list[dict[str, Any]]:
    results = []
    tts = TextToSpeech(voice=voice, speed=settings.tts.speed)
    tts.warmup()

    runs = []
    for text in TTS_TEXTS:
        start = time.time()
        first_chunk = 0.0
        samples = 0
        chunks = 0
        for _, _, audio in tts.synthesize_stream(text):
            chunks += 1
            if first_chunk == 0.0:
                first_chunk = time.time() - start
            samples += len(audio)
        total = time.time() - start
        audio_duration = samples / tts.sample_rate if samples else 0.0
        runs.append({
            "text": text,
            "first_chunk": first_chunk,
            "total": total,
            "audio_duration": audio_duration,
            "rtf": total / audio_duration if audio_duration else 0.0,
            "chunks": chunks,
        })

    results.append({
        "voice": voice,
        "summary": {
            "first_chunk": summarize([r["first_chunk"] for r in runs]),
            "rtf": summarize([r["rtf"] for r in runs]),
            "total": summarize([r["total"] for r in runs]),
        },
        "runs": runs,
    })

    del tts
    gc.collect()

    return results


def ensure_stt_fixtures() -> list[tuple[Path, str, list[str]]]:
    """Synthesize fixture wavs once (24kHz TTS resampled to 16kHz) and reuse them."""
    import scipy.io.wavfile as wavfile
    import scipy.signal

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    fixtures = []
    tts = None
    try:
        for name, text, keywords in STT_FIXTURES:
            path = FIXTURES_DIR / f"stt_{name}.wav"
            if not path.exists():
                if tts is None:
                    tts = TextToSpeech(voice=settings.tts.voice, speed=settings.tts.speed)
                    tts.warmup()
                chunks = [audio for _, _, audio in tts.synthesize_stream(text)]
                audio_24k = np.concatenate(chunks)
                audio_16k = scipy.signal.resample_poly(audio_24k, 2, 3).astype(np.float32)
                wavfile.write(path, 16000, audio_16k)
                print(f"Created fixture {path}")
            fixtures.append((path, text, keywords))
    finally:
        if tts is not None and hasattr(tts, "close"):
            tts.close()
    return fixtures


def load_fixture_audio(path: Path) -> np.ndarray:
    import scipy.io.wavfile as wavfile

    rate, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if rate != 16000:
        raise ValueError(f"Fixture {path} has sample rate {rate}, expected 16000")
    return audio


def benchmark_stt() -> list[dict[str, Any]]:
    from src.models.stt import SpeechToText

    fixtures = ensure_stt_fixtures()
    stt = SpeechToText()
    stt.warmup()

    runs = []
    for path, text, keywords in fixtures:
        audio = load_fixture_audio(path)
        start = time.time()
        transcript = stt.transcribe(audio)
        elapsed = time.time() - start
        lowered = transcript.lower()
        hits = [kw for kw in keywords if kw in lowered]
        runs.append({
            "fixture": path.name,
            "reference": text,
            "transcript": transcript,
            "time": elapsed,
            "audio_duration": len(audio) / 16000.0,
            "keyword_recall": len(hits) / len(keywords),
            "missing_keywords": [kw for kw in keywords if kw not in hits],
        })

    results = [{
        "model": stt.model_name,
        "summary": {
            "time": summarize([r["time"] for r in runs]),
            "keyword_recall": summarize([r["keyword_recall"] for r in runs]),
        },
        "runs": runs,
    }]

    del stt
    gc.collect()
    clear_mlx_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local voice chatbot latency")
    parser.add_argument("--llm", action="store_true", help="Run LLM benchmarks")
    parser.add_argument("--tts", action="store_true", help="Run TTS benchmarks")
    parser.add_argument("--stt", action="store_true", help="Run STT benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument(
        "--models",
        nargs="*",
        default=[settings.llm.model_name],
        help="LLM model names to benchmark",
    )
    parser.add_argument("--max-tokens", type=int, default=settings.llm.max_tokens)
    parser.add_argument("--voice", default=settings.tts.voice)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    run_llm = args.all or args.llm or not (args.llm or args.tts or args.stt)
    run_tts = args.all or args.tts
    run_stt = args.all or args.stt

    output: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "settings": {
            "llm_model": settings.llm.model_name,
            "max_tokens": args.max_tokens,
            "temperature": settings.llm.temperature,
            "top_p": settings.llm.top_p,
            "top_k": settings.llm.top_k,
            "tts_voice": args.voice,
        },
    }

    if run_stt:
        output["stt"] = benchmark_stt()
    if run_llm:
        output["llm"] = benchmark_llm(args.models, args.max_tokens)
    if run_tts:
        output["tts"] = benchmark_tts(args.voice)

    out_path = Path(args.output) if args.output else Path("logs") / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"\nSaved benchmark results to {out_path}")

    # Loading STT+LLM+TTS stacks in one process can SIGBUS in native library
    # teardown at interpreter exit (results are already saved by this point);
    # skip the teardown entirely
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
