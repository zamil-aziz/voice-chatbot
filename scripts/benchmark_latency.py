#!/usr/bin/env python3
"""
Benchmark local chatbot latency without using the microphone.

Examples:
    python -m scripts.benchmark_latency --llm
    python -m scripts.benchmark_latency --tts
    python -m scripts.benchmark_latency --all --models mlx-community/Qwen2.5-3B-Instruct-4bit
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


def benchmark_tts(devices: list[str], voice: str) -> list[dict[str, Any]]:
    results = []
    for device in devices:
        tts = TextToSpeech(voice=voice, speed=settings.tts.speed, device=device)
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
                "device": device,
                "first_chunk": first_chunk,
                "total": total,
                "audio_duration": audio_duration,
                "rtf": total / audio_duration if audio_duration else 0.0,
                "chunks": chunks,
            })

        results.append({
            "device": device,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local voice chatbot latency")
    parser.add_argument("--llm", action="store_true", help="Run LLM benchmarks")
    parser.add_argument("--tts", action="store_true", help="Run TTS benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
            settings.llm.model_name,
            "mlx-community/Qwen2.5-3B-Instruct-4bit",
            "mlx-community/Qwen2.5-7B-Instruct-4bit",
        ],
        help="LLM model names to benchmark",
    )
    parser.add_argument("--max-tokens", type=int, default=settings.llm.max_tokens)
    parser.add_argument("--tts-devices", nargs="*", default=["cpu", "mps"])
    parser.add_argument("--voice", default=settings.tts.voice)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    run_llm = args.all or args.llm or not (args.llm or args.tts)
    run_tts = args.all or args.tts

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

    if run_llm:
        output["llm"] = benchmark_llm(args.models, args.max_tokens)
    if run_tts:
        output["tts"] = benchmark_tts(args.tts_devices, args.voice)

    out_path = Path(args.output) if args.output else Path("logs") / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"\nSaved benchmark results to {out_path}")


if __name__ == "__main__":
    main()
