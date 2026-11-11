#!/usr/bin/env python3
"""Demo: load the corpus, index it, and run cross-language behavior searches.

    python demo.py

Shows that a Python sort query ranks the JavaScript sort above the
JavaScript fibonacci, that a JS fibonacci query finds the Python fibonacci,
and that plain-English sketches ("sorts a list") land on the right family.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from polyglot import default_corpus, search, describe_query  # noqa: E402

PY_SORT = '''
def bubble_sort(arr):
    n = len(arr)
    result = list(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
    return result
'''

JS_FIB = '''
function fibonacci(n) {
    if (n <= 1) { return n; }
    let a = 0;
    let b = 1;
    for (let i = 2; i <= n; i++) {
        let c = a + b;
        a = b;
        b = c;
    }
    return b;
}
'''


def show(query, corpus, top_k=4):
    print(f"\nQuery: {describe_query(query)}")
    for score, entry in search(query, corpus, top_k=top_k):
        print(f"  {score:.3f}  {entry.language:10s} {entry.name:15s} "
              f"ops={entry.ops} io={entry.io_examples[:1]}")


def main():
    corpus = default_corpus()
    print(f"Indexed {len(corpus)} functions "
          f"({len(corpus.by_language('python'))} Python, "
          f"{len(corpus.by_language('javascript'))} JavaScript)")
    for e in corpus:
        print(f"  {e.language:10s} {e.name:15s} "
              f"{len(e.io_examples)} I/O examples, e.g. {e.io_examples[:1]}")

    show(PY_SORT, corpus)          # cross-language: JS sort should rank top
    show(JS_FIB, corpus)           # JS query -> Python fibonacci should rank top
    show("sorts a list", corpus)   # sketch
    show("removes duplicates", corpus)
    show("reverses a string", corpus)


if __name__ == "__main__":
    main()
