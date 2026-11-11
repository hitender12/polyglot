# Architecture

## Idea in one paragraph

Syntax differs across languages; behavior does not. polyglot therefore
represents a function by *what it does*, in two complementary channels,
and searches in that representation space.

## Pipeline

```
  corpus/*.py ──┐
                ├─► extract.py ──► FunctionInfo(name, language, source,
  corpus/*.js ──┘                 │  ops=[...], io_examples=[...], arity)
                                  │
                                  ▼
                              embed.py ──► behavior vector (30 floats)
                                  │
                                  ▼
                    search.py: cosine(query_vec, entry_vec)
                                  │
                                  ▼
                         ranked [(score, FunctionInfo)]
```

## Components

### `polyglot/extract.py` — behavior extraction

- **Python:** `ast.parse` + a `NodeVisitor` that walks the tree in source
  order emitting abstract ops. `range()`/`len()` calls become `CALL`,
  `for`/`while` become `LOOP`, comparisons become `COMPARE`, tuple
  assignment `a[i], a[j] = a[j], a[i]` becomes `SWAP`, `.append()` becomes
  `APPEND`, subscripts become `INDEX`.
- **JavaScript:** the source is parsed by `js_interp.py` into a small
  dict-AST, which is walked the same way. An indexed assignment whose
  right-hand side indexes the same array (`arr[j] = arr[j+1]`) is
  emitted as `SWAP`, matching the Python tuple-swap heuristic.
- **Execution (both languages):** the function is run on a fixed probe
  set (`PROBES_1` for arity 1, `PROBES_2` for arity 2 — lists, strings,
  ints, and `(list, int)` pairs). A probe is kept only if the call
  returns without raising and does not return `None`; at most 6 examples
  are kept. Inputs are deep-copied per run so in-place mutation cannot
  leak between probes. Python runs via `exec` with a restricted
  `__builtins__` (no `import`/`open`/`eval`); JS runs in the bundled
  interpreter — never by shelling out to node.

### `polyglot/js_interp.py` — JS subset interpreter

Hand-written tokenizer (regex), recursive-descent parser (precedence
climbing for binary operators), and a tree-walking interpreter with
lexical `Env` chains, `_Return`/`_Break`/`_Continue` control-flow
exceptions, and JS semantics where they matter (`+` coerces to string
concatenation, `Math.floor/ceil/abs/max/min`, `.push()` returns the new
length, `.indexOf()` returns -1 when absent, `.slice()` with no args
copies the array). Supported subset is listed in its module docstring
and in the README; anything outside it raises `SyntaxError` /
`NotImplementedError` at parse/eval time.

### `polyglot/embed.py` — the behavior vector

30 dimensions: 12 op-histogram slots + 15 I/O features, with the I/O
block weighted ×2.5 (`IO_WEIGHT`) because *what a function does to data*
discriminates behaviors (sort vs dedup) better than *how often it loops*.

Op vocab: `ARITH ASSIGN BRANCH CALL COMPARE INDEX LOOP APPEND RETURN
SWAP SLICE CONCAT` (the last two are reserved for future extractors).

I/O features (`IO_NAMES`): input-type fractions (list/str/int),
output-type fractions (list/str/int/bool), output sortedness, output/input
length ratio, output uniqueness, output-is-reversed fraction,
multiset-preservation fraction, numeric growth (capped), and arity
one-hots. Each is an aggregate over the recorded I/O examples, so a
function with no successful probes gets a near-zero I/O block.

Similarity is plain cosine similarity (stdlib `math` only; numpy is
available but deliberately not required).

### `polyglot/search.py` — queries

- **Function query:** a `FunctionInfo`, or a source string. Source is
  detected as code if it contains `def ` / `function `, then extracted
  and embedded exactly like a corpus entry.
- **Sketch query:** anything else is looked up in `SKETCH_PROFILES` —
  a hand-written keyword → behavior-profile table. Longest keyword wins
  (`"binary search"` beats `"search"`, `"reverse a string"` beats
  `"reverse"`). The profile is a synthetic vector in the *same* 30-d
  space (an op histogram a function with that behavior would have, plus
  the I/O features it would exhibit), so sketch search is the same cosine
  ranking. Unknown sketches raise `ValueError` listing known behaviors.

### `polyglot/corpus.py` — indexing

`load_corpus(dir)` reads every `.py`/`.js` file, extracts it, and holds
`Corpus(entries)` in memory with precomputed vectors. A file that fails
to parse is skipped with a warning, not a crash. There is no server,
no persistence, and no ANN index — ranking is brute-force cosine.

## Data flow for a query

1. Query → vector (extract+embed, or sketch profile).
2. Cosine against every corpus vector.
3. Sort descending, return top_k `(score, FunctionInfo)` pairs.

## Design decisions and trade-offs

- **Histogram, not sequence:** chosen for simplicity and for
  cross-language robustness (statement order varies by idiom); the cost
  is that order-sensitive behaviors cannot be distinguished. A future
  version could add op-bigram features.
- **I/O weight 2.5:** tuned by hand on the 12-function corpus (sort vs
  dedup is the discriminating pair). This is a magic number; the
  evaluation plan in the README describes how to tune it properly.
- **Fixed probes over generated inputs:** keeps execution deterministic
  and fast, at the cost of coverage. Property-based / LLM-generated
  probes would be the next step.
- **No node dependency:** the JS interpreter is ~600 lines of extra
  code, but it makes `python demo.py` work on any machine with Python 3
  alone, which is the point of the demo.
