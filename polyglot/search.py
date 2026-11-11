"""Search: rank corpus functions by behavioral similarity to a query.

Two query forms:
  (a) a function -- a FunctionInfo, or Python/JS source code. The query is
      extracted and embedded exactly like a corpus entry, so a Python sort
      query finds the JS sort (same trace + same I/O) above a JS fibonacci.
  (b) a behavior sketch -- a natural-language-ish string such as
      "sorts a list" or "removes duplicates". This is mapped through
      SKETCH_PROFILES, a small hand-written keyword -> behavior-profile
      table (documented simplification: it is NOT an NLP model; it matches
      keywords and builds the vector a function with that behavior would
      have). Unknown sketches raise ValueError listing the known behaviors.
"""
from .embed import OP_VOCAB, OP_INDEX, IO_NAMES, N_IO, DIM, OP_WEIGHT, IO_WEIGHT, cosine, embed
from .extract import FunctionInfo, extract

# --- sketch profiles -------------------------------------------------------
# Each profile: (keywords, op histogram counts, io feature values).
# io keys are names from embed.IO_NAMES; unspecified io features are 0.
# Op counts are normalised inside _sketch_vector(), like a real histogram.
SKETCH_PROFILES = {
    "sort": {
        "keywords": ["sort", "sorted", "order", "ascending", "descending", "arrange"],
        "ops": {"LOOP": 4, "COMPARE": 3, "SWAP": 2, "INDEX": 3, "ASSIGN": 2, "RETURN": 1, "BRANCH": 1},
        "io": {"in_is_list": 1.0, "out_is_list": 1.0, "out_sorted": 1.0,
               "length_ratio": 1.0, "preserves_multiset": 1.0, "arity_1": 1.0},
    },
    "binary_search": {
        "keywords": ["binary search", "search sorted", "find in sorted", "lookup",
                     "binary", "search a sorted", "find an element"],
        "ops": {"LOOP": 2, "COMPARE": 3, "INDEX": 2, "BRANCH": 2, "ARITH": 2, "ASSIGN": 2, "RETURN": 1},
        "io": {"in_is_list": 1.0, "out_is_int": 1.0, "arity_2": 1.0},
    },
    "dedup": {
        "keywords": ["duplicate", "dedup", "unique", "remove duplicate", "deduplicate",
                     "distinct", "no repeats"],
        "ops": {"LOOP": 2, "BRANCH": 2, "COMPARE": 2, "APPEND": 2, "INDEX": 1, "CALL": 1, "RETURN": 1},
        "io": {"in_is_list": 1.0, "out_is_list": 1.0, "out_unique": 1.0,
               "length_ratio": 0.6, "arity_1": 1.0},
    },
    "fibonacci": {
        "keywords": ["fibonacci", "fib"],
        "ops": {"LOOP": 2, "ARITH": 3, "ASSIGN": 3, "BRANCH": 1, "RETURN": 1},
        "io": {"in_is_int": 1.0, "out_is_int": 1.0, "numeric_growth": 0.8, "arity_1": 1.0},
    },
    "reverse_string": {
        "keywords": ["reverse a string", "reverses a string", "reverse string",
                     "reverse text", "flip a string"],
        "ops": {"LOOP": 2, "INDEX": 1, "ASSIGN": 2, "ARITH": 1, "RETURN": 1},
        "io": {"in_is_str": 1.0, "out_is_str": 1.0, "out_reversed": 1.0,
               "length_ratio": 1.0, "preserves_multiset": 1.0, "arity_1": 1.0},
    },
    "reverse_list": {
        "keywords": ["reverse a list", "reverses a list", "reverse list"],
        "ops": {"LOOP": 2, "INDEX": 2, "ASSIGN": 2, "RETURN": 1},
        "io": {"in_is_list": 1.0, "out_is_list": 1.0, "out_reversed": 1.0,
               "length_ratio": 1.0, "preserves_multiset": 1.0, "arity_1": 1.0},
    },
    "reverse": {  # generic "reverse" / "reverses" -- string-flavoured default
        "keywords": ["reverse", "reverses", "backwards"],
        "ops": {"LOOP": 2, "INDEX": 1, "ASSIGN": 2, "ARITH": 1, "RETURN": 1},
        "io": {"in_is_str": 1.0, "out_is_str": 1.0, "out_reversed": 1.0,
               "length_ratio": 1.0, "preserves_multiset": 1.0, "arity_1": 1.0},
    },
    "sum": {
        "keywords": ["sum", "total", "add up", "add all"],
        "ops": {"LOOP": 2, "ARITH": 2, "ASSIGN": 2, "RETURN": 1},
        "io": {"in_is_list": 1.0, "out_is_int": 1.0, "arity_1": 1.0},
    },
}


def _sketch_vector(profile):
    vec = [0.0] * DIM
    total = sum(profile["ops"].values()) or 1
    for op, count in profile["ops"].items():
        vec[OP_INDEX[op]] = OP_WEIGHT * count / total
    for name, value in profile["io"].items():
        vec[len(OP_VOCAB) + IO_NAMES.index(name)] = IO_WEIGHT * value
    return vec


def match_sketch(text):
    """Return (behavior_name, vector) for a sketch string, or None."""
    lowered = text.lower()
    # Longest keyword first so "binary search" beats "search", and
    # "reverse a string" beats bare "reverse".
    candidates = []
    for name, prof in SKETCH_PROFILES.items():
        for kw in prof["keywords"]:
            if kw in lowered:
                candidates.append((len(kw), name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1], _sketch_vector(SKETCH_PROFILES[candidates[0][1]])


def _looks_like_code(query):
    q = query.strip()
    return ("def " in q or "function " in q or "=>" in q or
            q.startswith("def") or q.startswith("function"))


def _query_vector(query):
    """Return (vector, description) for any supported query form."""
    if isinstance(query, FunctionInfo):
        return embed(query), f"{query.name} ({query.language})"
    if isinstance(query, str):
        if _looks_like_code(query):
            lang = "javascript" if "function " in query else "python"
            info = extract(query, lang)
            return embed(info), f"{info.name} ({lang}, from source)"
        matched = match_sketch(query)
        if matched is not None:
            name, vec = matched
            return vec, f"sketch: {query!r} -> behavior {name!r}"
        raise ValueError(
            f"unknown behavior sketch {query!r}; known behaviors: "
            + ", ".join(sorted(SKETCH_PROFILES)))
    raise TypeError("query must be a FunctionInfo, source-code string, or sketch string")


def search(query, corpus, top_k=5):
    """Rank `corpus` (iterable of FunctionInfo) by behavioral similarity.

    Returns a list of (score, FunctionInfo) sorted by score descending.
    """
    qvec, _desc = _query_vector(query)
    scored = []
    for entry in corpus:
        scored.append((cosine(qvec, embed(entry)), entry))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:top_k]


def describe_query(query):
    """Human-readable description of what a query was interpreted as."""
    return _query_vector(query)[1]
