"""Behavioral embeddings.

A function becomes a fixed-length vector = weighted concatenation of

  1. a NORMALISED OPERATION HISTOGRAM over OP_VOCAB (what the code *does*,
     in language-neutral terms), and
  2. I/O MAPPING FEATURES measured from real executions (what the function
     *does to data*: input/output types, sortedness, length preservation,
     uniqueness, reversal, multiset preservation, numeric growth, arity).

Two functions in different languages that sort, dedup, or reverse land
close together because their traces and I/O agree, even though their
syntax shares nothing. Similarity is cosine similarity (pure stdlib;
numpy is not required).
"""
import math

OP_VOCAB = ["ARITH", "ASSIGN", "BRANCH", "CALL", "COMPARE", "INDEX",
            "LOOP", "APPEND", "RETURN", "SWAP", "SLICE", "CONCAT"]
OP_INDEX = {op: i for i, op in enumerate(OP_VOCAB)}

# I/O feature indices (after the op histogram)
IO_NAMES = [
    "in_is_list", "in_is_str", "in_is_int",      # 0-2  input types
    "out_is_list", "out_is_str", "out_is_int",   # 3-5  output types
    "out_sorted", "length_ratio", "out_unique",   # 6-8
    "out_reversed", "preserves_multiset",         # 9-10
    "numeric_growth", "arity_2", "arity_1",      # 11-13
    "out_is_bool",                                # 14
]
N_IO = len(IO_NAMES)
DIM = len(OP_VOCAB) + N_IO

OP_WEIGHT = 1.0
IO_WEIGHT = 2.5  # behaviour (what it does to data) outweighs raw op counts


def _is_sorted(xs):
    return all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))


def io_features(info):
    """Aggregate I/O features over the function's recorded examples."""
    f = [0.0] * N_IO
    ex = info.io_examples
    if not ex:
        f[12] = 1.0 if info.arity == 1 else 0.0
        f[13] = 1.0 if info.arity == 2 else 0.0
        return f
    n = len(ex)

    def frac(pred):
        return sum(1 for args, out in ex if pred(args, out)) / n

    f[0] = frac(lambda a, o: isinstance(a[0], list))
    f[1] = frac(lambda a, o: isinstance(a[0], str))
    f[2] = frac(lambda a, o: isinstance(a[0], int) and not isinstance(a[0], bool))
    f[3] = frac(lambda a, o: isinstance(o, list))
    f[4] = frac(lambda a, o: isinstance(o, str))
    f[5] = frac(lambda a, o: isinstance(o, int) and not isinstance(o, bool))
    f[14] = frac(lambda a, o: isinstance(o, bool))

    list_ex = [(a, o) for a, o in ex if isinstance(o, list) and isinstance(a[0], (list, str))]
    if list_ex:
        f[6] = sum(1 for _, o in list_ex if _is_sorted(o)) / len(list_ex)
        ratios = [len(o) / len(a[0]) for a, o in list_ex if len(a[0]) > 0]
        f[7] = sum(ratios) / len(ratios) if ratios else 0.0
        f[8] = sum(1 for _, o in list_ex if len(set(o)) == len(o)) / len(list_ex)
        f[9] = sum(1 for a, o in list_ex
                   if list(o) == list(reversed(a[0]))) / len(list_ex)
        f[10] = sum(1 for a, o in list_ex
                    if isinstance(a[0], list) and sorted(o) == sorted(a[0])) / len(list_ex)
    str_ex = [(a, o) for a, o in ex if isinstance(o, str) and isinstance(a[0], str)]
    if str_ex:
        # reversal / length features also apply to strings
        f[7] = max(f[7], sum(len(o) / len(a[0]) for a, o in str_ex if len(a[0]) > 0) / len(str_ex))
        f[9] = max(f[9], sum(1 for a, o in str_ex if o == a[0][::-1]) / len(str_ex))
        f[10] = max(f[10], sum(1 for a, o in str_ex if sorted(o) == sorted(a[0])) / len(str_ex))
    int_ex = [(a, o) for a, o in ex
              if isinstance(o, int) and isinstance(a[0], int)
              and not isinstance(o, bool)]
    if int_ex:
        growth = [min(o / a[0], 5.0) / 5.0 for a, o in int_ex if a[0] > 1]
        f[11] = sum(growth) / len(growth) if growth else 0.0
    f[12] = 1.0 if info.arity == 2 else 0.0
    f[13] = 1.0 if info.arity == 1 else 0.0
    # length_ratio is a ratio, not a 0..1 feature in the same sense; cap it.
    f[7] = min(f[7], 1.0)
    return f


def embed(info):
    """Return the behavior vector for a FunctionInfo (list of floats)."""
    vec = [0.0] * DIM
    total = len(info.ops) or 1
    for op in info.ops:
        if op in OP_INDEX:
            vec[OP_INDEX[op]] += OP_WEIGHT / total
    io = io_features(info)
    for i, v in enumerate(io):
        vec[len(OP_VOCAB) + i] = IO_WEIGHT * v
    return vec


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def similarity(info_a, info_b):
    return cosine(embed(info_a), embed(info_b))
