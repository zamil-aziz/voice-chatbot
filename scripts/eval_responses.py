#!/usr/bin/env python3
"""
Evaluation script for testing LLM response quality.
Tests for conversational naturalness, emotional warmth, and anti-robotic behavior.

Usage:
    python -m scripts.eval_responses
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

from rich.console import Console
from rich.table import Table

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.llm import LanguageModel
from config.settings import settings

console = Console()

# Banned phrases that indicate robotic/formal responses
BANNED_PHRASES = [
    "certainly",
    "absolutely",
    "of course",
    "definitely",
    "i'd be happy to",
    "i understand",
    "rest assured",
    "great question",
    "that's a great point",
    "is there anything else",
    "as an ai",
    "as a language model",
    "i'm sorry to hear that",
    "i apologize",
    "please let me know",
    "feel free to",
    "don't hesitate",
]

# Test prompts covering different emotional contexts
TEST_PROMPTS = [
    # Emotional - sad/stressed
    ("I had a really terrible day at work today", "empathy"),
    ("I'm so stressed about this deadline", "empathy"),
    ("My dog passed away yesterday", "empathy"),

    # Emotional - excited/happy
    ("I just got engaged!", "excitement"),
    ("I got the promotion!", "excitement"),
    ("Guess what? I'm gonna be a dad!", "excitement"),

    # Simple questions
    ("What's the weather like?", "factual"),
    ("What time is it?", "factual"),
    ("How do I make pasta?", "informational"),

    # Casual conversation
    ("Hey, what's up?", "greeting"),
    ("I'm bored, got any ideas?", "casual"),
    ("Tell me something interesting", "casual"),
]


def check_banned_phrases(response: str) -> List[str]:
    """Check if response contains any banned phrases."""
    found = []
    response_lower = response.lower()
    for phrase in BANNED_PHRASES:
        if phrase in response_lower:
            found.append(phrase)
    return found


def check_response_length(response: str) -> Tuple[int, str]:
    """Check if response length is appropriate (1-3 sentences ideal)."""
    # Count sentences (rough approximation)
    sentences = len(re.findall(r'[.!?]+', response))
    words = len(response.split())

    if sentences <= 2 and words <= 30:
        return sentences, "good"
    elif sentences <= 3 and words <= 50:
        return sentences, "okay"
    else:
        return sentences, "too_long"


def check_warmth_indicators(response: str) -> List[str]:
    """Check for warm, natural language indicators."""
    warmth_indicators = [
        r"\boh\b", r"\baw\b", r"\bhmm\b", r"\booh\b",
        r"\bwow\b", r"\byay\b", r"\bugh\b", r"\bwhoa\b",
        r"\.{3}", r"!", r"\?",  # ellipses, exclamations, questions
        r"that's (so |really )?", r"how (are|was|did)",
    ]
    found = []
    response_lower = response.lower()
    for pattern in warmth_indicators:
        if re.search(pattern, response_lower):
            found.append(pattern)
    return found


def evaluate_response(prompt: str, response: str, prompt_type: str) -> Dict:
    """Evaluate a single response."""
    banned = check_banned_phrases(response)
    sentence_count, length_rating = check_response_length(response)
    warmth = check_warmth_indicators(response)

    # Score: 0-10
    score = 10
    issues = []

    # Deduct for banned phrases
    if banned:
        score -= len(banned) * 2
        issues.append(f"Banned phrases: {banned}")

    # Deduct for length issues
    if length_rating == "too_long":
        score -= 2
        issues.append(f"Too long ({sentence_count} sentences)")

    # Deduct for lack of warmth in emotional contexts
    if prompt_type in ["empathy", "excitement"] and len(warmth) < 2:
        score -= 2
        issues.append("Lacks emotional warmth")

    return {
        "prompt": prompt,
        "response": response,
        "prompt_type": prompt_type,
        "score": max(0, score),
        "banned_phrases": banned,
        "sentence_count": sentence_count,
        "length_rating": length_rating,
        "warmth_indicators": len(warmth),
        "issues": issues,
    }


def run_evaluation():
    """Run full evaluation suite."""
    console.print("\n[bold cyan]Loading LLM for evaluation...[/bold cyan]\n")

    llm = LanguageModel(
        model_name=settings.llm.model_name,
        max_tokens=settings.llm.max_tokens,
        temperature=settings.llm.temperature,
        system_prompt=settings.llm.system_prompt,
    )

    results = []

    console.print("[bold]Running evaluation prompts...[/bold]\n")

    for prompt, prompt_type in TEST_PROMPTS:
        console.print(f"[dim]Testing: {prompt[:50]}...[/dim]")

        # Clear history for each test
        llm.clear_history()

        response = llm.generate(prompt)
        result = evaluate_response(prompt, response, prompt_type)
        results.append(result)

        # Show result inline
        color = "green" if result["score"] >= 8 else "yellow" if result["score"] >= 5 else "red"
        console.print(f"  [{color}]Score: {result['score']}/10[/{color}] - {response[:60]}...")

    # Summary table
    console.print("\n[bold cyan]Evaluation Summary[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Prompt Type", style="dim")
    table.add_column("Avg Score")
    table.add_column("Issues")

    # Group by type
    by_type = {}
    for r in results:
        t = r["prompt_type"]
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(r)

    total_score = 0
    for prompt_type, type_results in by_type.items():
        avg_score = sum(r["score"] for r in type_results) / len(type_results)
        total_score += avg_score
        all_issues = [i for r in type_results for i in r["issues"]]

        color = "green" if avg_score >= 8 else "yellow" if avg_score >= 5 else "red"
        table.add_row(
            prompt_type,
            f"[{color}]{avg_score:.1f}/10[/{color}]",
            ", ".join(list(set(all_issues))[:3]) if all_issues else "[green]None[/green]"
        )

    console.print(table)

    overall = total_score / len(by_type)
    console.print(f"\n[bold]Overall Score: {overall:.1f}/10[/bold]")

    # Save results
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"eval_{timestamp}.json"

    with open(log_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "overall_score": overall,
            "results": results,
            "settings": {
                "model": settings.llm.model_name,
                "temperature": settings.llm.temperature,
            }
        }, f, indent=2)

    console.print(f"\n[dim]Results saved to: {log_file}[/dim]")

    return results


if __name__ == "__main__":
    run_evaluation()
