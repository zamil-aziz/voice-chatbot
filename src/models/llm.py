"""
Language Model module using MLX (Qwen family).
Optimized for Apple Silicon with streaming support.
"""

import time
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Generator, Optional, List, Dict

from rich.console import Console
from rich.markup import escape

from config.settings import settings
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import make_prompt_cache

console = Console()


class LanguageModel:
    """Language model using MLX (supports Qwen, Llama, etc.)."""

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen3.5-4B-MLX-4bit",
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        history_turns: int = 6,
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

        # Conversation prompt cache: holds a stable rendered-history prefix
        # of the current prompt (never generation output), so each turn only
        # the newest messages are processed. Kept pristine by generating on a
        # clone; this also works for hybrid models whose linear-attention
        # caches cannot be trimmed. The stable prefix is discovered as the
        # common prefix of consecutive turns' prompts, which needs no extra
        # chat-template render.
        self._conversation_cache: Optional[list] = None
        self._cached_token_ids: List[int] = []
        self._last_prompt_tokens: Optional[List[int]] = None

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
            from mlx_lm import load
            return load(self.model_name)

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(do_load)
                self.model, self.tokenizer = future.result(
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

    def _render_messages(self, messages: List[Dict[str, str]], add_generation_prompt: bool) -> List[int]:
        """Render chat messages into token ids with the model's template."""
        template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
        }
        if "qwen3" in self.model_name.lower():
            template_kwargs["enable_thinking"] = self.enable_thinking

        try:
            tokens = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("enable_thinking", None)
            tokens = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        return list(tokens)

    def _build_messages(self, user_content: str) -> List[Dict[str, str]]:
        """System prompt (with today's date) plus history plus the new user turn."""
        date_str = datetime.now().strftime("%B %d, %Y")
        system_with_date = f"Today is {date_str}.\n\n{self.system_prompt}"
        return (
            [{"role": "system", "content": system_with_date}]
            + self.conversation_history
            + [{"role": "user", "content": user_content}]
        )

    @staticmethod
    def _common_prefix_len(a: List[int], b: List[int]) -> int:
        """Length of the longest common prefix of two token id lists."""
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _prefill(self, tokens: List[int], chunk_size: int = 512) -> None:
        """Process tokens into the conversation cache without generating."""
        import mlx.core as mx

        for i in range(0, len(tokens), chunk_size):
            chunk = mx.array(tokens[i:i + chunk_size])[None]
            self.model(chunk, cache=self._conversation_cache)
        mx.eval([c.state for c in self._conversation_cache])

    def _advance_conversation_cache(self, prompt_tokens: List[int]) -> int:
        """Align and grow the conversation cache for this turn's prompt.

        The cache must hold a prefix of prompt_tokens; on divergence (e.g. a
        history trim changed the prefix) it is dropped and rebuilt. It then
        grows to the prefix this prompt shares with the previous turn's
        prompt: that shared region is exactly the stably rendered history
        (the volatile tail - RAG-injected user content plus the generation
        prompt - differs between consecutive renders and is never cached).
        Returns the number of tokens this turn did not have to re-process.
        """
        if self._cached_token_ids:
            prefix = self._common_prefix_len(self._cached_token_ids, prompt_tokens)
            if prefix < len(self._cached_token_ids):
                self._conversation_cache = None
                self._cached_token_ids = []
        reused = len(self._cached_token_ids)

        if self._last_prompt_tokens is not None:
            # Cap so the generation suffix is never empty
            stable = min(
                self._common_prefix_len(self._last_prompt_tokens, prompt_tokens),
                len(prompt_tokens) - 1,
            )
            if stable > len(self._cached_token_ids):
                if self._conversation_cache is None:
                    self._conversation_cache = make_prompt_cache(self.model)
                self._prefill(prompt_tokens[len(self._cached_token_ids):stable])
                self._cached_token_ids = prompt_tokens[:stable]

        self._last_prompt_tokens = prompt_tokens
        return reused

    def _clone_conversation_cache(self) -> list:
        """Cheap copy of the conversation cache for one generation.

        Generation appends the volatile prompt tail and its own output to the
        cache it runs on; cloning keeps the conversation cache pristine so it
        stays a pure extension target next turn. State arrays can be shared:
        MLX arrays are updated functionally, and KVCache writes past the
        clone's offset always trigger buffer expansion first. Only the list
        container of ArraysCache-style states needs copying.
        """
        clone = make_prompt_cache(self.model)
        for src, dst in zip(self._conversation_cache, clone):
            state = src.state
            dst.state = list(state) if isinstance(state, list) else state
            meta = getattr(src, "meta_state", None)
            if meta is not None:
                dst.meta_state = meta
        return clone

    @staticmethod
    def _build_user_content(user_message: str, context: Optional[List[str]]) -> str:
        """Build the prompt content for a user turn, injecting RAG notes if any.

        Only the returned prompt content carries the notes; conversation
        history must store the original user message so the context window
        never fills up with retrieval instructions.
        """
        if not context:
            return user_message

        context_text = "\n".join(context)
        return f"""[Background notes about the person you're talking to that MAY be relevant:
{context_text}

Use them only if they genuinely help with this message; otherwise ignore them silently.]

{user_message}"""

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

    # Extra turns tolerated beyond history_turns before trimming. Trimming
    # invalidates the conversation prompt cache (the rendered prefix changes),
    # so trimming in batches keeps cache rebuilds rare instead of every turn
    # once the cap is reached.
    HISTORY_TRIM_SLACK_TURNS = 4

    def _trim_history(self) -> None:
        """Keep conversation history bounded for lower prompt-processing latency."""
        max_messages = max(0, self.history_turns) * 2
        if max_messages == 0:
            self.conversation_history.clear()
            return
        slack_messages = self.HISTORY_TRIM_SLACK_TURNS * 2
        if len(self.conversation_history) > max_messages + slack_messages:
            self.conversation_history = self.conversation_history[-max_messages:]

    def commit_turn(
        self, user_message: str, assistant_response: str, interrupted: bool = False
    ) -> None:
        """Append a completed (or interrupted) turn to conversation history.

        The pipeline owns when a turn is committed so barge-in can record
        only what was actually spoken before the interruption.
        """
        if interrupted:
            assistant_response = f"{assistant_response} [interrupted by the user]".strip()
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_response}
        )
        self._trim_history()

    def generate(self, user_message: str, context: Optional[List[str]] = None) -> str:
        """
        Generate a response to the user message.

        Args:
            user_message: The user's input text
            context: Optional list of relevant context strings (from RAG)

        Returns:
            The assistant's response
        """
        start = time.time()

        chunks = list(self.generate_stream(user_message, context=context))
        assistant_response = self.clean_response_text("".join(chunks))

        elapsed = time.time() - start
        console.print(f"[dim]LLM ({elapsed:.2f}s): {escape(assistant_response[:50])}...[/dim]")

        self.commit_turn(user_message, assistant_response)

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

        from mlx_lm import stream_generate

        # Bring the conversation cache up to date, then generate on a clone
        # so the cache itself stays pristine for the next turn.
        # RAG notes go into this turn's prompt only, never into history.
        tokenize_start = time.time()
        messages = self._build_messages(self._build_user_content(user_message, context))
        prompt_tokens = self._render_messages(messages, add_generation_prompt=True)
        reused_tokens = self._advance_conversation_cache(prompt_tokens)

        suffix_tokens = prompt_tokens[len(self._cached_token_ids):]
        if self._conversation_cache is not None and suffix_tokens:
            generation_cache = self._clone_conversation_cache()
        else:
            reused_tokens = 0
            suffix_tokens = prompt_tokens
            generation_cache = make_prompt_cache(self.model)
        tokenize_time = time.time() - tokenize_start

        first_token_time = None
        token_count = 0
        last_response = None
        gen_start = time.time()

        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=suffix_tokens,
            max_tokens=self.max_tokens,
            sampler=self._sampler,
            prompt_cache=generation_cache,
        ):
            last_response = response

            # stream_generate can emit a final empty segment with finish metadata.
            text = response.text if hasattr(response, 'text') else str(response)
            if not text:
                continue

            token_count += 1
            if first_token_time is None:
                first_token_time = time.time() - gen_start

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
            "cache_reused_tokens": reused_tokens,
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
            f"{tokens_per_sec:.1f} tok/s ({token_count} tokens), "
            f"cache reuse {reused_tokens}/{len(prompt_tokens)}[/dim]"
        )

    def warmup(self) -> None:
        """Warm up the model to avoid cold-start latency on first real inference."""
        if self.model is None:
            return

        console.print("[dim]Warming up LLM...[/dim]")
        start = time.time()
        from mlx_lm import stream_generate

        # No prompt_cache here: warm-up must not pollute the conversation cache
        for _ in stream_generate(
            self.model,
            self.tokenizer,
            prompt="Hi",
            max_tokens=1,
            sampler=self._sampler,
        ):
            pass
        console.print(f"[dim]LLM warm-up done in {time.time() - start:.2f}s[/dim]")

    def clear_history(self) -> None:
        """Clear conversation history and the prompt cache built on it."""
        self.conversation_history = []
        self._conversation_cache = None
        self._cached_token_ids = []
        self._last_prompt_tokens = None
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
