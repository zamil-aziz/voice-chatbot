"""
Language Model module using Llama via MLX.
Optimized for Apple Silicon with streaming support.
"""

import time
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Generator, Optional, List, Dict

from rich.console import Console

from config.settings import settings
from mlx_lm.sample_utils import make_sampler

console = Console()


class LanguageModel:
    """Language model using MLX (supports Qwen, Llama, etc.)."""

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        max_tokens: int = 96,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        history_turns: int = 4,
        enable_thinking: bool = False,
        system_prompt: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.history_turns = history_turns
        self.enable_thinking = enable_thinking
        self.system_prompt = system_prompt or self._default_system_prompt()

        self.model = None
        self.tokenizer = None
        self.conversation_history: List[Dict[str, str]] = []
        self._sampler = None  # Cached sampler instance
        self.last_stream_stats: Dict[str, float] = {}

        self._load_model()

    def _default_system_prompt(self) -> str:
        return """You are a helpful, friendly voice assistant.
Keep your responses concise and conversational - remember this will be spoken aloud.
Aim for 1-3 sentences unless more detail is specifically requested.
Be natural and warm in your tone."""

    def _load_model(self) -> None:
        """Load LLM model with timeout. Downloads if not cached."""
        console.print(f"[yellow]Loading LLM: {self.model_name}[/yellow]")
        start = time.time()

        def do_load():
            from mlx_lm import load, generate
            model, tokenizer = load(self.model_name, tokenizer_config={"eos_token": "<|im_end|>"})
            return model, tokenizer, generate

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(do_load)
                self.model, self.tokenizer, self._generate_fn = future.result(
                    timeout=settings.model_load_timeout
                )

            # Create cached sampler instance
            self._sampler = make_sampler(
                temp=self.temperature,
                top_p=self.top_p,
                min_p=self.min_p,
                top_k=self.top_k,
            )

            console.print(
                f"[green]LLM ready in {time.time() - start:.2f}s[/green]"
            )
        except FuturesTimeoutError:
            raise RuntimeError(
                f"LLM model loading timed out after {settings.model_load_timeout}s"
            )
        except ImportError as e:
            console.print(f"[red]Failed to import mlx_lm: {e}[/red]")
            console.print("[yellow]Run: pip install mlx-lm[/yellow]")
            raise

    def _format_messages(self, user_message: str) -> str:
        """Format messages using the model's chat template."""
        # Add current date so LLM knows what day it is
        date_str = datetime.now().strftime("%B %d, %Y")
        system_with_date = f"Today is {date_str}.\n\n{self.system_prompt}"

        messages = [{"role": "system", "content": system_with_date}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_message})

        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if "qwen3" in self.model_name.lower():
            template_kwargs["enable_thinking"] = self.enable_thinking

        try:
            return self.tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **template_kwargs)

    @staticmethod
    def clean_response_text(text: str) -> str:
        """Normalize model output for spoken playback and conversation history."""
        text = re.sub(r"<\s*think\b[^>]*>.*?</\s*think\s*>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<\s*think\b[^>]*>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"</?\s*think\b[^>]*>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"`([^`]*)`", r"\1", text)
        text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"(?<=[.!?])\s+[-*+]\s+", " ", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _trim_history(self) -> None:
        """Keep conversation history bounded for lower prompt-processing latency."""
        max_messages = max(0, self.history_turns) * 2
        if max_messages == 0:
            self.conversation_history.clear()
        elif len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]

    def generate(self, user_message: str, context: Optional[List[str]] = None) -> str:
        """
        Generate a response to the user message.

        Args:
            user_message: The user's input text
            context: Optional list of relevant context strings (from RAG)

        Returns:
            The assistant's response
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Inject RAG context if provided
        if context:
            context_text = "\n".join(context)
            user_message = f"""[You know these things about the person you're talking to:
{context_text}

Weave this knowledge into your response - mention specific details you know about them to show you remember and care. Make them feel known.]

User: {user_message}"""

        start = time.time()

        # Format the prompt
        prompt = self._format_messages(user_message)

        # Generate response using cached sampler
        response = self._generate_fn(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=self._sampler,
            verbose=False,
        )

        # Extract just the assistant's response
        # The generate function returns the full text, we need to strip the prompt
        assistant_response = self.clean_response_text(response)

        elapsed = time.time() - start
        console.print(f"[dim]LLM ({elapsed:.2f}s): {assistant_response[:50]}...[/dim]")

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_response}
        )

        self._trim_history()

        return assistant_response

    def generate_stream(self, user_message: str, context: Optional[List[str]] = None) -> Generator[str, None, None]:
        """
        Generate a streaming response to the user message.

        Args:
            user_message: The user's input text
            context: Optional list of relevant context strings (from RAG)

        Yields:
            Chunks of the assistant's response
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Inject RAG context if provided
        if context:
            context_text = "\n".join(context)
            user_message = f"""[You know these things about the person you're talking to:
{context_text}

Weave this knowledge into your response - mention specific details you know about them to show you remember and care. Make them feel known.]

User: {user_message}"""

        from mlx_lm import stream_generate

        # Measure tokenization time
        tokenize_start = time.time()
        prompt = self._format_messages(user_message)
        tokenize_time = time.time() - tokenize_start

        full_response = ""
        first_token_time = None
        token_count = 0
        last_response = None
        gen_start = time.time()

        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=self._sampler,
        ):
            last_response = response

            # stream_generate can emit a final empty segment with finish metadata.
            text = response.text if hasattr(response, 'text') else str(response)
            if not text:
                continue

            token_count += 1
            if first_token_time is None:
                first_token_time = time.time() - gen_start

            full_response += text
            yield text

        # Log detailed timing
        total_gen_time = time.time() - gen_start
        tokens_per_sec = token_count / total_gen_time if total_gen_time > 0 else 0
        first_token_time = first_token_time or 0
        self.last_stream_stats = {
            "tokenize": tokenize_time,
            "first_token": first_token_time,
            "generation_total": total_gen_time,
            "generation_tps": tokens_per_sec,
            "generation_tokens": token_count,
        }
        if last_response is not None:
            self.last_stream_stats.update({
                "prompt_tokens": getattr(last_response, "prompt_tokens", 0),
                "prompt_tps": getattr(last_response, "prompt_tps", 0.0),
                "peak_memory_gb": getattr(last_response, "peak_memory", 0.0),
            })
        console.print(
            f"[dim]LLM detail: tok={tokenize_time*1000:.0f}ms, "
            f"first={first_token_time*1000:.0f}ms, "
            f"{tokens_per_sec:.1f} tok/s ({token_count} tokens)[/dim]"
        )

        cleaned_response = self.clean_response_text(full_response)

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append(
            {"role": "assistant", "content": cleaned_response}
        )

        self._trim_history()

    def warmup(self) -> None:
        """Warm up the model to avoid cold-start latency on first real inference."""
        if self.model is None:
            return

        console.print("[dim]Warming up LLM...[/dim]")
        start = time.time()
        _ = self._generate_fn(
            self.model,
            self.tokenizer,
            prompt="Hi",
            max_tokens=1,
            sampler=self._sampler,
            verbose=False,
        )
        console.print(f"[dim]LLM warm-up done in {time.time() - start:.2f}s[/dim]")

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        console.print("[dim]Conversation history cleared[/dim]")


# Quick test
if __name__ == "__main__":
    llm = LanguageModel()

    # Test generation
    response = llm.generate("Hello! What's your name?")
    console.print(f"[green]Response: {response}[/green]")

    # Test follow-up (uses conversation history)
    response = llm.generate("What did I just ask you?")
    console.print(f"[green]Response: {response}[/green]")
