"""Tests for polyglot. Run:  python -m pytest tests/ -v
(or: python -m unittest discover -s tests -v)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polyglot import (  # noqa: E402
    default_corpus, extract, search, cosine, embed, match_sketch)
from polyglot.js_interp import run_js_function  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

PY_SORT = open(os.path.join(ROOT, "corpus", "sort.py")).read()
JS_SORT = open(os.path.join(ROOT, "corpus", "sort.js")).read()
JS_FIB = open(os.path.join(ROOT, "corpus", "fibonacci.js")).read()
PY_FIB = open(os.path.join(ROOT, "corpus", "fibonacci.py")).read()


def rank_of(results, name, language=None):
    for i, (_score, entry) in enumerate(results):
        if entry.name == name and (language is None or entry.language == language):
            return i
    return None


class TestCrossLanguageSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = default_corpus()

    def test_corpus_loads_all_twelve(self):
        self.assertEqual(len(self.corpus), 12)
        self.assertEqual(len(self.corpus.by_language("python")), 6)
        self.assertEqual(len(self.corpus.by_language("javascript")), 6)

    def test_python_sort_query_ranks_js_sort_above_js_fibonacci(self):
        results = search(PY_SORT, self.corpus, top_k=12)
        self.assertLess(rank_of(results, "bubbleSort", "javascript"),
                        rank_of(results, "fibonacci", "javascript"))

    def test_python_sort_query_ranks_js_sort_second_only_behind_itself(self):
        # The query is identical to the Python sort in the corpus, so that
        # entry is (correctly) #1 at 1.0; the JS sort must be #2, ahead of
        # every unrelated function.
        results = search(PY_SORT, self.corpus, top_k=12)
        self.assertEqual(results[0][1].name, "bubble_sort")
        self.assertEqual(results[1][1].name, "bubbleSort")
        self.assertEqual(results[1][1].language, "javascript")
        self.assertGreater(results[1][0], 0.9)

    def test_js_fibonacci_query_ranks_python_fibonacci_second(self):
        # Same self-match logic: JS fib is #1, Python fib must be #2 and
        # well ahead of unrelated functions such as sum.
        results = search(JS_FIB, self.corpus, top_k=12)
        self.assertEqual(results[0][1].name, "fibonacci")
        self.assertEqual(results[1][1].name, "fibonacci")
        self.assertEqual(results[1][1].language, "python")
        self.assertGreater(results[1][0], 0.9)
        self.assertLess(rank_of(results, "fibonacci", "python"),
                        rank_of(results, "sumList", "javascript"))

    def test_same_language_equivalent_ranks_first(self):
        # Python fib query: the Python fib itself (identical behavior) is #1.
        results = search(PY_FIB, self.corpus, top_k=12)
        self.assertEqual(results[0][1].name, "fibonacci")
        self.assertEqual(results[0][1].language, "python")
        self.assertAlmostEqual(results[0][0], 1.0, places=6)

    def test_unseen_sort_variant_ranks_js_sort_first(self):
        # An insertion sort that is NOT in the corpus: its nearest neighbor
        # must still be a sort, in the other language, above fibonacci.
        insertion_sort = '''
def insertion_sort(arr):
    result = list(arr)
    for i in range(1, len(result)):
        key = result[i]
        j = i - 1
        while j >= 0 and result[j] > key:
            result[j + 1] = result[j]
            j = j - 1
        result[j + 1] = key
    return result
'''
        results = search(insertion_sort, self.corpus, top_k=12)
        self.assertIn(results[0][1].name, ("bubble_sort", "bubbleSort"))
        self.assertLess(rank_of(results, "bubbleSort", "javascript"),
                        rank_of(results, "fibonacci", "javascript"))

    def test_self_similarity_is_one(self):
        info = extract(PY_SORT, "python")
        self.assertAlmostEqual(cosine(embed(info), embed(info)), 1.0, places=6)


class TestExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = default_corpus()

    def _by(self, name, lang):
        return next(e for e in self.corpus if e.name == name and e.language == lang)

    @staticmethod
    def _out(info, args):
        for a, o in info.io_examples:
            if a == args:
                return o
        raise KeyError(args)

    def test_io_examples_produced_for_every_corpus_function(self):
        for entry in self.corpus:
            self.assertGreater(len(entry.io_examples), 0, entry.name)

    def test_python_sort_io_correct(self):
        info = self._by("bubble_sort", "python")
        self.assertIn(([3, 1, 2],), [a for a, _ in info.io_examples])
        self.assertEqual(self._out(info, ([3, 1, 2],)), [1, 2, 3])

    def test_js_sort_io_correct_via_interpreter(self):
        self.assertEqual(run_js_function(JS_SORT, "bubbleSort", [[3, 1, 2]]), [1, 2, 3])
        info = self._by("bubbleSort", "javascript")
        self.assertEqual(self._out(info, ([3, 1, 2],)), [1, 2, 3])

    def test_binary_search_io_correct_both_languages(self):
        py = self._by("binary_search", "python")
        js = self._by("binarySearch", "javascript")
        self.assertEqual(self._out(py, ([1, 2, 3, 4, 5], 3)), 2)
        self.assertEqual(self._out(js, ([1, 2, 3, 4, 5], 3)), 2)

    def test_dedup_io_correct_both_languages(self):
        py = self._by("dedup", "python")
        js = self._by("dedup", "javascript")
        self.assertEqual(self._out(py, ([1, 2, 2, 3, 1],)), [1, 2, 3])
        self.assertEqual(self._out(js, ([1, 2, 2, 3, 1],)), [1, 2, 3])

    def test_fibonacci_io_correct_both_languages(self):
        py = self._by("fibonacci", "python")
        js = self._by("fibonacci", "javascript")
        self.assertEqual(self._out(py, (10,)), 55)
        self.assertEqual(self._out(js, (10,)), 55)

    def test_reverse_string_io_correct_both_languages(self):
        py = self._by("reverse_string", "python")
        js = self._by("reverseString", "javascript")
        self.assertEqual(self._out(py, ("hello",)), "olleh")
        self.assertEqual(self._out(js, ("hello",)), "olleh")

    def test_js_interpreter_needs_no_node(self):
        # Direct interpreter call only; nothing here shells out.
        self.assertEqual(run_js_function(JS_FIB, "fibonacci", [10]), 55)


class TestSketches(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = default_corpus()

    def test_sort_sketch_returns_sort_family_first(self):
        results = search("sorts a list", self.corpus, top_k=12)
        self.assertIn(results[0][1].name, ("bubble_sort", "bubbleSort"))

    def test_dedup_sketch_returns_dedup_first(self):
        results = search("removes duplicates", self.corpus, top_k=12)
        self.assertEqual(results[0][1].name, "dedup")

    def test_reverse_sketch_returns_reverse_family_first(self):
        results = search("reverses a string", self.corpus, top_k=12)
        self.assertIn(results[0][1].name, ("reverse_string", "reverseString"))

    def test_fibonacci_sketch_returns_fibonacci_first(self):
        results = search("fibonacci", self.corpus, top_k=12)
        self.assertEqual(results[0][1].name, "fibonacci")

    def test_binary_search_sketch_returns_binary_search_first(self):
        results = search("binary search", self.corpus, top_k=12)
        self.assertIn(results[0][1].name, ("binary_search", "binarySearch"))

    def test_unknown_sketch_raises(self):
        with self.assertRaises(ValueError):
            search("does quantum teleportation", self.corpus)

    def test_top_k_respected(self):
        self.assertEqual(len(search("sorts a list", self.corpus, top_k=3)), 3)


if __name__ == "__main__":
    unittest.main()
