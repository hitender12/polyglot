"""Corpus loading / indexing: read .py and .js files from a directory,
extract each function (trace + executed I/O), embed it, and hold the index
in memory. Indexing is just extraction + embedding; there is no server.
"""
import os

from .embed import embed
from .extract import extract


class Corpus:
    def __init__(self, entries):
        self.entries = list(entries)
        self.vectors = {id(e): embed(e) for e in self.entries}

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def by_name(self, name):
        return [e for e in self.entries if e.name == name]

    def by_language(self, language):
        return [e for e in self.entries if e.language == language]


def load_corpus(directory):
    """Load every .py/.js file in `directory` as one corpus entry each."""
    entries = []
    for fname in sorted(os.listdir(directory)):
        path = os.path.join(directory, fname)
        if not os.path.isfile(path):
            continue
        if fname.endswith(".py"):
            lang = "python"
        elif fname.endswith(".js"):
            lang = "javascript"
        else:
            continue
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
        try:
            entries.append(extract(source, lang, path=path))
        except Exception as exc:  # keep a broken file from killing the index
            print(f"warning: skipping {path}: {exc}")
    return Corpus(entries)


def default_corpus():
    here = os.path.dirname(os.path.abspath(__file__))
    return load_corpus(os.path.join(here, "..", "corpus"))
