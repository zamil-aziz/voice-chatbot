"""
RAG (Retrieval-Augmented Generation) module for personal notes.
Uses a simple text file for storage and sentence-transformers for embeddings.
"""

import time
from pathlib import Path
from typing import Optional
import numpy as np

from rich.console import Console

from config.settings import settings

console = Console()


class NotesRAG:
    """
    Simple RAG system for personal notes.

    Loads notes from a text file (one per line) and keeps
    normalized embeddings in memory for fast semantic search.
    """

    def __init__(self, notes_file: Optional[str] = None):
        """
        Initialize the RAG system.

        Args:
            notes_file: Path to the notes text file (one note per line).
                        Defaults to settings.rag.notes_file.
        """
        self.notes_file = Path(notes_file or settings.rag.notes_file)
        self.embedder = None
        self.notes: list[str] = []
        self.embeddings: np.ndarray | None = None
        self._loaded = False

    def _load(self) -> None:
        """Load the embedder and notes from file."""
        if self._loaded:
            return

        console.print("[yellow]Loading RAG components...[/yellow]")
        start = time.time()

        from sentence_transformers import SentenceTransformer

        # Load embedding model (small, fast, good quality)
        self.embedder = SentenceTransformer(settings.rag.embedder_model)

        # Load notes from file
        self._load_notes_from_file()

        self._loaded = True
        console.print(f"[green]RAG ready in {time.time() - start:.2f}s ({len(self.notes)} notes)[/green]")

    def _load_notes_from_file(self) -> None:
        """Load and embed notes from the text file."""
        if not self.notes_file.exists():
            console.print(f"[dim]No notes file found at {self.notes_file}[/dim]")
            self.notes = []
            self.embeddings = None
            return

        # Read notes (one per line, skip empty lines and comments)
        with open(self.notes_file, "r", encoding="utf-8") as f:
            self.notes = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]

        if not self.notes:
            self.embeddings = None
            return

        # Embed all notes once, L2-normalized so search is a single dot product
        console.print(f"[dim]Embedding {len(self.notes)} notes...[/dim]")
        embeddings = self.embedder.encode(self.notes, convert_to_numpy=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = embeddings / np.maximum(norms, 1e-12)

    def reload(self) -> None:
        """Reload notes from file (call after editing notes.txt)."""
        if self.embedder is None:
            self._load()
        else:
            self._load_notes_from_file()
            console.print(f"[green]Reloaded {len(self.notes)} notes[/green]")

    def _similarities(self, query: str) -> np.ndarray:
        """Cosine similarity between the query and every note."""
        query_embedding = self.embedder.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding / max(np.linalg.norm(query_embedding), 1e-12)
        return self.embeddings @ query_embedding

    def search(
        self,
        query: str,
        n_results: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> list[str]:
        """
        Search for notes relevant to the query.

        Args:
            query: The search query (user's speech or question)
            n_results: Maximum number of results (default settings.rag.n_results)
            min_similarity: Minimum cosine similarity to include a result
                            (default settings.rag.min_similarity)

        Returns:
            Relevant note texts, best match first. Empty when nothing is
            relevant enough; callers should then leave the prompt untouched.
        """
        self._load()

        if not self.notes or self.embeddings is None:
            return []

        if n_results is None:
            n_results = settings.rag.n_results
        if min_similarity is None:
            min_similarity = settings.rag.min_similarity

        similarities = self._similarities(query)
        top_indices = np.argsort(similarities)[::-1][:n_results]
        results = [self.notes[i] for i in top_indices if similarities[i] >= min_similarity]

        if results:
            console.print(f"[dim]RAG found {len(results)} relevant notes[/dim]")

        return results

    def count(self) -> int:
        """Get the number of notes loaded."""
        self._load()
        return len(self.notes)


# Quick test
if __name__ == "__main__":
    rag = NotesRAG()

    console.print("\n[bold]Testing search:[/bold]")

    queries = [
        "What does my wife like to do?",
        "Tell me about Sarah",
        "Do I have any pets?",
    ]

    for query in queries:
        console.print(f"\n[cyan]Query:[/cyan] {query}")
        for note in rag.search(query):
            console.print(f"  {note}")

    console.print(f"\n[bold]Total notes: {rag.count()}[/bold]")
