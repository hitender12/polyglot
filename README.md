# polyglot — search code by behavior, across languages

Code search today is lexical: `grep`, AST matching, token embeddings. The
same algorithm written in Python and in JavaScript looks unrelated to all
of them, even though it *does the same thing*.

**polyglot searches by behavior instead.** A function is embedded from two
things that survive translation between languages:

1. a **symbolic execution trace** — the ordered list of language-neutral
   abstract operations the function performs (`LOOP`, `COMPARE`, `SWAP`,
   `APPEND`, `INDEX`, `ARITH`, `BRANCH`, `CALL`, `RETURN`, …), and
2. **real I/O behavior** — the function is *actually executed* on probe
   inputs, and the input/output mapping is measured (output sortedness,
   length preservation, uniqueness, reversal, multiset preservation,
   numeric growth, input/output types, arity).

Functions that sort cluster together whether they are written in Python or
JavaScript; a Fibonacci in either language lands far from both. Similarity
is cosine similarity over that behavior vector.

## Quickstart

```bash
python demo.py
python -m unittest discover -s tests -v   # 22 tests, all passing
```

```python
from polyglot import default_corpus, search

corpus = default_corpus()  # 12 functions: 6 behaviors x Python + JS

# (a) Query with a function -- cross-language by construction.
results = search(open("corpus/sort.py").read(), corpus, top_k=3)
for score, fn in results:
    print(f"{score:.3f}  {fn.language:10s} {fn.name}")

# (b) Query with a behavior sketch (plain English, keyword-mapped).
results = search("removes duplicates", corpus, top_k=3)
```

Demo output (abridged, verified by running `python demo.py`):

```
Indexed 12 functions (6 Python, 6 JavaScript)
Query: bubble_sort (python, from source)
  1.000  python     bubble_sort
  0.999  javascript bubbleSort      <- cross-language equivalent, #2
  0.888  javascript dedup
Query: fibonacci (javascript, from source)
  1.000  javascript fibonacci
  0.999  python     fibonacci       <- cross-language equivalent, #2
Query: sketch: 'sorts a list' -> behavior 'sort'
  0.939  javascript bubbleSort
  0.939  python     bubble_sort
Query: sketch: 'removes duplicates' -> behavior 'dedup'
  0.937  python     dedup
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md). In short:

| Module | Role |
|---|---|
| `polyglot/extract.py` | Python `ast` traces + restricted-`exec` execution; JS traces + execution via the bundled interpreter |
| `polyglot/js_interp.py` | Hand-written JS tokenizer / parser / interpreter for the supported subset (no node, no subprocess) |
| `polyglot/embed.py` | Behavior vector: normalised op histogram + I/O features; cosine similarity |
| `polyglot/search.py` | `search(query, corpus, top_k)`; function queries and sketch queries |
| `polyglot/corpus.py` | Loads `corpus/*.py` / `*.js` into an in-memory index |
| `corpus/` | Equivalent implementations in Python + JS: sort, binary search, dedup, Fibonacci, string reverse, sum |

## Honest status / limitations

This is a **working proof of concept on a tiny, curated corpus**, not a
production code-search engine. What is real, and what is simplified:

- **Real:** the trace extraction, the JS interpreter, the execution-based
  I/O probing, the embeddings, cosine search, the demo, and all 22 tests
  (run and passing — see below).
- **JS subset only.** Supported: function declarations, `const`/`let`/`var`,
  `return`, `if`/`else`, `for`/`while`, `break`/`continue`, array literals,
  indexing, `.length`, `.push()`, `.indexOf()`, `.includes()`, `.slice()`,
  `Math.floor/ceil/abs/max/min`, ternary, `++`/`--`, compound assignment.
  Not supported: classes, objects, arrow functions, closures, async,
  regex, methods beyond the list above, `switch`, `try/catch`.
- **Sketch queries are a keyword table, not NLP.** `"sorts a list"` works
  because `sort` is in `SKETCH_PROFILES` in `search.py`; an unknown
  sketch raises `ValueError` listing the known behaviors. There is no
  language model and no learned mapping.
- **Op histogram, not trace order.** The embedding uses a *histogram* of
  ops, so two functions with the same op counts in a different order look
  identical. Sequence / data-flow information is not yet encoded.
- **Fixed probe inputs.** I/O examples come from a fixed probe set
  (`PROBES_1` / `PROBES_2` in `extract.py`), not generated test cases.
  A function whose behavior only shows on other inputs will be
  mis-embedded, and a function that raises on every probe gets no I/O
  features at all. JavaScript's loose typing also lets probes "succeed"
  nonsensically — e.g. the JS `reverseString` on a list input returns
  `"213"` — which pollutes its I/O features slightly.
- **Python execution is sandbox-ish, not a sandbox.** `exec` runs with a
  restricted `__builtins__` (no `import`, `open`, `eval`), but this is
  not a security boundary; do not run untrusted code with it.
- **Tiny corpus, no scale story.** 12 functions, in-memory, brute-force
  cosine over all entries. No indexing structure, no persistence, no
  GitHub-scale ingestion.

## Evaluation plan

What would turn this into a real evaluation (not yet done):

1. **Held-out equivalents:** write N new implementations per behavior in
   both languages (different algorithms — e.g. insertion sort, merge
   sort, recursive Fibonacci, set-based dedup) and measure recall@k:
   does the true equivalent rank in the top k?
2. **Confusability matrix:** sort vs dedup vs reverse are close cousins
   (all loop + compare + build a collection); report per-behavior
   precision, and tune the I/O-feature weights (`IO_WEIGHT` in
   `embed.py`) against that matrix instead of by hand.
3. **Ablation:** op-histogram-only vs I/O-only vs combined, to show both
   signals actually contribute.
4. **Robustness:** functions that raise on probes, empty functions,
   functions with side effects, and JS outside the subset — measure
   graceful-degradation rates.
5. **Scale smoke test:** index 10k synthetic functions and measure
   query latency (brute-force cosine is O(corpus); an ANN index would
   be needed beyond that).

## Tests

```bash
python -m unittest discover -s tests -v
# Ran 22 tests in 0.036s -- OK
```

pytest is **not** installed in this environment, so the suite is
`unittest`-based (the test file header says so); it also runs under
pytest if pytest is available.

## License

MIT — see [LICENSE](LICENSE).
