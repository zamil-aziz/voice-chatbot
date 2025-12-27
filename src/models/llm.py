"""
Language Model module using Llama via MLX.
Optimized for Apple Silicon with streaming support.
"""

import time
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
        model_name: str = "mlx-community/Qwen2.5-7B-Instruct-4bit",
        max_tokens: int = 256,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or self._default_system_prompt()

        self.model = None
        self.tokenizer = None
        self.conversation_history: List[Dict[str, str]] = []
        self._sampler = None  # Cached sampler instance

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
            self._sampler = make_sampler(temp=self.temperature)

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

        # Apply chat template
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

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

Use this naturally in conversation - reference what you know to make them feel understood and known, but don't force it.]

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
        assistant_response = response.strip()

        elapsed = time.time() - start
        console.print(f"[dim]LLM ({elapsed:.2f}s): {assistant_response[:50]}...[/dim]")

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_response}
        )

        # Keep conversation history bounded (6 turns = 12 messages for faster inference)
        if len(self.conversation_history) > 12:
            self.conversation_history = self.conversation_history[-12:]

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

Use this naturally in conversation - reference what you know to make them feel understood and known, but don't force it.]

User: {user_message}"""

        from mlx_lm import stream_generate

        # Measure tokenization time
        tokenize_start = time.time()
        prompt = self._format_messages(user_message)
        tokenize_time = time.time() - tokenize_start

        full_response = ""
        first_token_time = None
        token_count = 0
        gen_start = time.time()

        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=self._sampler,
        ):
            token_count += 1
            if first_token_time is None:
                first_token_time = time.time() - gen_start

            # stream_generate yields response objects with .text attribute
            text = response.text if hasattr(response, 'text') else str(response)
            full_response += text
            yield text

        # Log detailed timing
        total_gen_time = time.time() - gen_start
        tokens_per_sec = token_count / total_gen_time if total_gen_time > 0 else 0
        console.print(
            f"[dim]LLM detail: tok={tokenize_time*1000:.0f}ms, "
            f"first={first_token_time*1000:.0f}ms, "
            f"{tokens_per_sec:.1f} tok/s ({token_count} tokens)[/dim]"
        )

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append(
            {"role": "assistant", "content": full_response.strip()}
        )

        # Keep conversation history bounded (6 turns = 12 messages for faster inference)
        if len(self.conversation_history) > 12:
            self.conversation_history = self.conversation_history[-12:]

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
