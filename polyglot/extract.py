"""Behavior extraction: parse Python (ast) and a JS subset (js_interp),
normalize both to a language-neutral operation trace, and obtain I/O
examples by ACTUALLY EXECUTING the function on probe inputs.

Python is executed via exec() in a namespace with a restricted __builtins__
(no import/open/eval). JS is executed by the bundled interpreter in
js_interp.py -- never by shelling out to node.
"""
import ast
import copy
from dataclasses import dataclass, field

from .js_interp import parse_js, JSInterpreter

# Canonical probe inputs, tried in order per arity. A probe is kept only
# when the function returns without raising. Inputs are deep-copied per run
# so in-place mutation cannot leak between probes.
PROBES_1 = [[3, 1, 2], [5, 4, 3, 2, 1], [1, 2, 2, 3, 1], "hello",
            "racecar", 10, 5, 0, [5, 4, 3], "abc", [2, 2, 2]]
PROBES_2 = [([1, 2, 3, 4, 5], 3), ([3, 1, 2], 2), ([1, 2, 3], 1),
            ("hello", "l"), ([10, 20, 30], 20)]
MAX_IO_EXAMPLES = 6

SAFE_BUILTINS = {
    "len": len, "range": range, "min": min, "max": max, "sum": sum,
    "abs": abs, "enumerate": enumerate, "list": list, "str": str,
    "int": int, "float": float, "bool": bool, "sorted": sorted,
    "set": set, "dict": dict, "tuple": tuple, "zip": zip,
    "isinstance": isinstance, "round": round, "reversed": reversed,
}


@dataclass
class FunctionInfo:
    name: str
    language: str  # "python" | "javascript"
    source: str
    ops: list = field(default_factory=list)          # ordered abstract ops
    io_examples: list = field(default_factory=list)  # [(args_tuple, output)]
    arity: int = 1
    path: str = ""

    def __repr__(self):
        return f"FunctionInfo({self.name!r}, {self.language}, ops={len(self.ops)}, io={len(self.io_examples)})"


# ---------------------------------------------------------------- Python
def _py_ops(tree):
    """Walk the AST in source order, emitting abstract ops."""
    ops = []

    class V(ast.NodeVisitor):
        def visit_For(self, node):
            ops.append("LOOP")
            self.generic_visit(node)

        def visit_While(self, node):
            ops.append("LOOP")
            self.generic_visit(node)

        def visit_If(self, node):
            ops.append("BRANCH")
            self.generic_visit(node)

        def visit_Compare(self, node):
            ops.append("COMPARE")
            self.generic_visit(node)

        def visit_BinOp(self, node):
            ops.append("ARITH")
            self.generic_visit(node)

        def visit_Call(self, node):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "append":
                ops.append("APPEND")
            elif isinstance(func, ast.Attribute) and func.attr in ("sort", "reverse"):
                ops.append("CALL")
            else:
                ops.append("CALL")
            self.generic_visit(node)

        def visit_Subscript(self, node):
            ops.append("INDEX")
            self.generic_visit(node)

        def visit_Return(self, node):
            ops.append("RETURN")
            self.generic_visit(node)

        def visit_Assign(self, node):
            # tuple swap: a[i], a[j] = a[j], a[i]
            if isinstance(node.targets[0], ast.Tuple) and isinstance(node.value, ast.Tuple):
                ops.append("SWAP")
            else:
                ops.append("ASSIGN")
            self.generic_visit(node)

        def visit_AugAssign(self, node):
            ops.append("ASSIGN")
            self.generic_visit(node)

        def visit_ListComp(self, node):
            ops.append("LOOP")
            self.generic_visit(node)

    V().visit(tree)
    return ops


def extract_python(source, name=None, path=""):
    tree = ast.parse(source)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if fn is None:
        raise ValueError("no top-level function definition found in Python source")
    name = name or fn.name
    ops = _py_ops(tree)
    info = FunctionInfo(name=name, language="python", source=source,
                        ops=ops, arity=len(fn.args.args), path=path)
    info.io_examples = _run_python(source, name, info.arity)
    return info


def _run_python(source, name, arity):
    ns = {"__builtins__": dict(SAFE_BUILTINS)}
    try:
        exec(compile(source, "<polyglot>", "exec"), ns)
    except Exception:
        return []
    fn = ns.get(name)
    if not callable(fn):
        return []
    return _probe(fn, arity)


# ---------------------------------------------------------------- JS
def _js_ops(program):
    """Walk the JS AST in source order, emitting the same abstract ops."""
    ops = []

    def walk(node):
        if isinstance(node, list):
            for x in node:
                walk(x)
            return
        if not isinstance(node, dict):
            return
        t = node.get("type")
        if t == "For" or t == "While":
            ops.append("LOOP")
        elif t == "If":
            ops.append("BRANCH")
        elif t == "Ternary":
            ops.append("BRANCH")
        elif t == "Binary":
            if node["op"] in ("<", ">", "<=", ">=", "==", "!=", "===", "!=="):
                ops.append("COMPARE")
            elif node["op"] in ("+", "-", "*", "/", "%"):
                ops.append("ARITH")
        elif t == "MethodCall":
            ops.append("APPEND" if node["method"] == "push" else "CALL")
        elif t == "Call":
            ops.append("CALL")
        elif t == "Index":
            ops.append("INDEX")
        elif t == "Member":
            pass  # .length etc. carry no behavior op
        elif t == "Return":
            ops.append("RETURN")
        elif t == "Assign":
            target, val = node["target"], node["value"]
            # indexed swap: arr[j] = arr[j+1] (RHS indexes the same array)
            if target.get("type") == "Index" and _indexes_same_array(target, val):
                ops.append("SWAP")
            else:
                ops.append("ASSIGN")
        elif t == "VarDecl":
            ops.append("ASSIGN")
        for v in node.values():
            walk(v)

    walk(program)
    return ops


def _indexes_same_array(target, val):
    """True if `val` is (or contains) an Index into the same base as target."""
    def base(n):
        while isinstance(n, dict) and n.get("type") == "Index":
            n = n["obj"]
        return n if isinstance(n, dict) else None

    tb = base(target)
    if tb is None:
        return False

    def contains(n):
        if isinstance(n, dict):
            if n.get("type") == "Index" and base(n) == tb:
                return True
            return any(contains(v) for v in n.values())
        if isinstance(n, list):
            return any(contains(v) for v in n)
        return False

    return contains(val)


def extract_javascript(source, name=None, path=""):
    program = parse_js(source)
    fn = next((s for s in program["body"] if s["type"] == "FunctionDecl"), None)
    if fn is None:
        raise ValueError("no function declaration found in JS source")
    name = name or fn["name"]
    ops = _js_ops(program)
    info = FunctionInfo(name=name, language="javascript", source=source,
                        ops=ops, arity=len(fn["params"]), path=path)
    info.io_examples = _run_js(source, name, info.arity)
    return info


def _run_js(source, name, arity):
    try:
        interp = JSInterpreter(parse_js(source))
    except Exception:
        return []
    return _probe(lambda *a: interp.call(name, list(a)), arity)


# ---------------------------------------------------------------- shared
def _probe(fn, arity):
    probes = PROBES_1 if arity == 1 else PROBES_2 if arity == 2 else []
    out = []
    for p in probes:
        args = (p,) if arity == 1 else tuple(p)
        try:
            result = fn(*copy.deepcopy(args))
        except Exception:
            continue
        if result is None:
            continue
        # JS division can return floats like 2.0 for ints; normalise.
        result = _normalise(result)
        out.append((args, result))
        if len(out) >= MAX_IO_EXAMPLES:
            break
    return out


def _normalise(v):
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, list):
        return [_normalise(x) for x in v]
    return v


def extract(source, language, name=None, path=""):
    """Dispatch on language: 'python'/'py' or 'javascript'/'js'."""
    lang = language.lower()
    if lang in ("python", "py"):
        return extract_python(source, name, path)
    if lang in ("javascript", "js"):
        return extract_javascript(source, name, path)
    raise ValueError(f"unsupported language {language!r}")
