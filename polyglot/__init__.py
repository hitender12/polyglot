"""polyglot -- search code by behavior, across languages.

Functions are embedded from symbolic execution traces + real I/O behavior,
not syntax, so semantically identical code clusters together regardless of
language. See README.md and ARCHITECTURE.md.
"""
from .extract import FunctionInfo, extract, extract_python, extract_javascript
from .embed import embed, cosine, similarity, OP_VOCAB
from .search import search, match_sketch, describe_query
from .corpus import Corpus, load_corpus, default_corpus

__all__ = [
    "FunctionInfo", "extract", "extract_python", "extract_javascript",
    "embed", "cosine", "similarity", "OP_VOCAB",
    "search", "match_sketch", "describe_query",
    "Corpus", "load_corpus", "default_corpus",
]

__version__ = "0.1.0"
