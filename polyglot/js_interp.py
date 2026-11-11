"""A small, dependency-free JavaScript subset: tokenizer, parser, interpreter.

Supported subset (documented limitation, see README):
  - function declarations, const/let/var, return, if/else, for, while,
    break, continue
  - expressions: literals, identifiers, array literals, indexing,
    .length, .push(), .indexOf(), .includes(), .slice(), Math.floor/ceil/
    abs/max/min, binary operators (+ - * / % < > <= >= == === != !==
    && ||), unary (!, -), ternary (?:), ++/--, += -= *= /= %=
  - string concatenation with +, string indexing, string .length
No classes, objects, arrow functions, closures over mutable state beyond
function scope, async, regex, or DOM. This interpreter exists so JS can be
*executed* for I/O examples without shelling out to node.
"""
import math
import re

KEYWORDS = {"function", "const", "let", "var", "return", "if", "else",
            "for", "while", "break", "continue", "true", "false", "null",
            "undefined"}

TOKEN_RE = re.compile(r"""
    \s*(?:
    (?P<comment>//[^\n]*|/\*.*?\*/)
  | (?P<string>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
  | (?P<number>\d+(?:\.\d+)?)
  | (?P<ident>[A-Za-z_$][A-Za-z0-9_$]*)
  | (?P<op>===|!==|==|!=|<=|>=|&&|\|\||\+\+|--|\+=|-=|\*=|/=|%=|[+\-*/%<>=!(){}[\];,.:?])
    )
""", re.VERBOSE | re.DOTALL)


class JSToken:
    def __init__(self, kind, value):
        self.kind = kind
        self.value = value

    def __repr__(self):
        return f"Tok({self.kind},{self.value!r})"


def tokenize(src):
    tokens = []
    pos = 0
    while pos < len(src):
        while pos < len(src) and src[pos].isspace():
            pos += 1
        if pos >= len(src):
            break
        m = TOKEN_RE.match(src, pos)
        if not m or m.end() == pos:
            raise SyntaxError(f"JS tokenizer: unexpected char {src[pos]!r} at {pos}")
        pos = m.end()
        if m.group("comment") is not None:
            continue
        if m.group("string") is not None:
            raw = m.group("string")
            tokens.append(JSToken("string", raw[1:-1]
                                 .replace("\\n", "\n").replace("\\t", "\t")
                                 .replace("\\'", "'").replace('\\"', '"')
                                 .replace("\\\\", "\\")))
        elif m.group("number") is not None:
            s = m.group("number")
            tokens.append(JSToken("number", float(s) if "." in s else int(s)))
        elif m.group("ident") is not None:
            tokens.append(JSToken("ident", m.group("ident")))
        elif m.group("op") is not None:
            tokens.append(JSToken("op", m.group("op")))
    tokens.append(JSToken("eof", None))
    return tokens


class Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.i = 0

    def peek(self, k=0):
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def next(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, value):
        t = self.next()
        if t.value != value:
            raise SyntaxError(f"JS parser: expected {value!r}, got {t.value!r}")
        return t

    def accept(self, value):
        if self.peek().value == value:
            self.next()
            return True
        return False

    # ---- program ----
    def parse_program(self):
        stmts = []
        while self.peek().kind != "eof":
            stmts.append(self.parse_statement())
        return {"type": "Program", "body": stmts}

    def parse_function(self):
        self.expect("function")
        name = self.next().value
        self.expect("(")
        params = []
        if self.peek().value != ")":
            params.append(self.next().value)
            while self.accept(","):
                params.append(self.next().value)
        self.expect(")")
        body = self.parse_block()
        return {"type": "FunctionDecl", "name": name, "params": params, "body": body}

    def parse_block(self):
        self.expect("{")
        stmts = []
        while self.peek().value != "}":
            stmts.append(self.parse_statement())
        self.expect("}")
        return {"type": "Block", "body": stmts}

    def parse_statement(self):
        t = self.peek()
        if t.value == "function":
            return self.parse_function()
        if t.value in ("const", "let", "var"):
            self.next()
            name = self.next().value
            init = None
            if self.accept("="):
                init = self.parse_expression()
            self.accept(";")
            return {"type": "VarDecl", "name": name, "init": init}
        if t.value == "return":
            self.next()
            val = None
            if self.peek().value != ";":
                val = self.parse_expression()
            self.accept(";")
            return {"type": "Return", "value": val}
        if t.value == "if":
            self.next()
            self.expect("(")
            cond = self.parse_expression()
            self.expect(")")
            then = self.parse_statement()
            els = None
            if self.accept("else"):
                els = self.parse_statement()
            return {"type": "If", "cond": cond, "then": then, "else": els}
        if t.value == "for":
            self.next()
            self.expect("(")
            init = None
            if self.peek().value != ";":
                if self.peek().value in ("let", "var", "const"):
                    init = self.parse_statement_no_semi()
                else:
                    init = self.parse_expression()
            self.expect(";")
            cond = None
            if self.peek().value != ";":
                cond = self.parse_expression()
            self.expect(";")
            update = None
            if self.peek().value != ")":
                update = self.parse_expression()
            self.expect(")")
            body = self.parse_statement()
            return {"type": "For", "init": init, "cond": cond, "update": update, "body": body}
        if t.value == "while":
            self.next()
            self.expect("(")
            cond = self.parse_expression()
            self.expect(")")
            body = self.parse_statement()
            return {"type": "While", "cond": cond, "body": body}
        if t.value == "break":
            self.next(); self.accept(";")
            return {"type": "Break"}
        if t.value == "continue":
            self.next(); self.accept(";")
            return {"type": "Continue"}
        if t.value == "{":
            return self.parse_block()
        # expression statement
        expr = self.parse_expression()
        self.accept(";")
        return {"type": "ExprStmt", "expr": expr}

    def parse_statement_no_semi(self):
        # var decl inside for(;;) header, without trailing ';'
        self.next()  # let/var/const
        name = self.next().value
        init = None
        if self.accept("="):
            init = self.parse_expression()
        return {"type": "VarDecl", "name": name, "init": init}

    # ---- expressions (precedence climbing) ----
    def parse_expression(self):
        return self.parse_assign()

    def parse_assign(self):
        node = self.parse_ternary()
        if self.peek().value in ("=", "+=", "-=", "*=", "/=", "%="):
            op = self.next().value
            rhs = self.parse_assign()
            return {"type": "Assign", "op": op, "target": node, "value": rhs}
        return node

    def parse_ternary(self):
        node = self.parse_binary(0)
        if self.accept("?"):
            a = self.parse_expression()
            self.expect(":")
            b = self.parse_expression()
            return {"type": "Ternary", "cond": node, "then": a, "else": b}
        return node

    BIN_PREC = {"||": 1, "&&": 2, "==": 3, "!=": 3, "===": 3, "!==": 3,
                "<": 4, ">": 4, "<=": 4, ">=": 4, "+": 5, "-": 5,
                "*": 6, "/": 6, "%": 6}

    def parse_binary(self, min_prec):
        left = self.parse_unary()
        while True:
            op = self.peek().value
            prec = self.BIN_PREC.get(op)
            if prec is None or prec < min_prec:
                break
            self.next()
            right = self.parse_binary(prec + 1)
            left = {"type": "Binary", "op": op, "left": left, "right": right}
        return left

    def parse_unary(self):
        if self.peek().value in ("!", "-"):
            op = self.next().value
            return {"type": "Unary", "op": op, "expr": self.parse_unary()}
        if self.peek().value in ("++", "--"):
            op = self.next().value
            return {"type": "PreInc", "op": op, "expr": self.parse_unary()}
        return self.parse_postfix()

    def parse_postfix(self):
        node = self.parse_primary()
        while True:
            if self.peek().value == ".":
                self.next()
                prop = self.next().value
                if self.peek().value == "(":
                    self.next()
                    args = self.parse_args()
                    node = {"type": "MethodCall", "obj": node, "method": prop, "args": args}
                else:
                    node = {"type": "Member", "obj": node, "prop": prop}
            elif self.peek().value == "[":
                self.next()
                idx = self.parse_expression()
                self.expect("]")
                node = {"type": "Index", "obj": node, "index": idx}
            elif self.peek().value == "(" and node.get("type") in ("Identifier", "Member"):
                self.next()
                args = self.parse_args()
                node = {"type": "Call", "callee": node, "args": args}
            elif self.peek().value in ("++", "--"):
                op = self.next().value
                node = {"type": "PostInc", "op": op, "expr": node}
            else:
                break
        return node

    def parse_args(self):
        args = []
        if self.peek().value != ")":
            args.append(self.parse_expression())
            while self.accept(","):
                args.append(self.parse_expression())
        self.expect(")")
        return args

    def parse_primary(self):
        t = self.next()
        if t.kind == "number":
            return {"type": "Literal", "value": t.value}
        if t.kind == "string":
            return {"type": "Literal", "value": t.value}
        if t.value in ("true", "false"):
            return {"type": "Literal", "value": t.value == "true"}
        if t.value in ("null", "undefined"):
            return {"type": "Literal", "value": None}
        if t.value == "(":
            node = self.parse_expression()
            self.expect(")")
            return node
        if t.value == "[":
            elems = []
            if self.peek().value != "]":
                elems.append(self.parse_expression())
                while self.accept(","):
                    elems.append(self.parse_expression())
            self.expect("]")
            return {"type": "ArrayLit", "elems": elems}
        if t.kind == "ident":
            return {"type": "Identifier", "name": t.value}
        raise SyntaxError(f"JS parser: unexpected token {t.value!r}")


def parse_js(src):
    return Parser(tokenize(src)).parse_program()


class _Return(Exception):
    def __init__(self, value):
        self.value = value


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class Env:
    def __init__(self, parent=None):
        self.vars = {}
        self.parent = parent

    def get(self, name):
        if name in self.vars:
            return self.vars[name]
        if self.parent is not None:
            return self.parent.get(name)
        raise NameError(f"JS: undefined variable {name!r}")

    def set(self, name, value):
        env = self
        while env is not None:
            if name in env.vars:
                env.vars[name] = value
                return
            env = env.parent
        self.vars[name] = value

    def define(self, name, value):
        self.vars[name] = value


def _js_truthy(v):
    if v is None or v is False:
        return False
    if isinstance(v, (int, float)) and v == 0:
        return False
    if isinstance(v, str) and v == "":
        return False
    return True


def _js_eq(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b if isinstance(a, bool) and isinstance(b, bool) else a == b
    return a == b


class JSInterpreter:
    """Interpret a parsed JS program; `call(name, args)` invokes a function."""

    def __init__(self, program):
        self.program = program
        self.global_env = Env()
        self.functions = {}
        for stmt in program["body"]:
            if stmt["type"] == "FunctionDecl":
                self.functions[stmt["name"]] = stmt
                self.global_env.define(stmt["name"], stmt)

    def call(self, name, args, _depth=0):
        if _depth > 500:
            raise RecursionError("JS: max call depth")
        fn = self.functions.get(name)
        if fn is None:
            raise NameError(f"JS: no function {name!r}")
        env = Env(self.global_env)
        for pname, val in zip(fn["params"], args):
            env.define(pname, val)
        try:
            self.exec_block(fn["body"], env)
        except _Return as r:
            return r.value
        return None

    def exec_block(self, block, env):
        for stmt in block["body"]:
            self.exec_stmt(stmt, env)

    def exec_stmt(self, stmt, env):
        t = stmt["type"]
        if t == "Block":
            self.exec_block(stmt, Env(env))
        elif t == "VarDecl":
            env.define(stmt["name"],
                       self.eval(stmt["init"], env) if stmt["init"] is not None else None)
        elif t == "Return":
            raise _Return(self.eval(stmt["value"], env) if stmt["value"] is not None else None)
        elif t == "If":
            if _js_truthy(self.eval(stmt["cond"], env)):
                self.exec_stmt(stmt["then"], env)
            elif stmt["else"] is not None:
                self.exec_stmt(stmt["else"], env)
        elif t == "For":
            env2 = Env(env)
            if stmt["init"] is not None:
                if stmt["init"]["type"] == "VarDecl":
                    env2.define(stmt["init"]["name"],
                                self.eval(stmt["init"]["init"], env2)
                                if stmt["init"]["init"] is not None else None)
                else:
                    self.eval(stmt["init"], env2)
            while stmt["cond"] is None or _js_truthy(self.eval(stmt["cond"], env2)):
                try:
                    self.exec_stmt(stmt["body"], env2)
                except _Break:
                    break
                except _Continue:
                    pass
                if stmt["update"] is not None:
                    self.eval(stmt["update"], env2)
        elif t == "While":
            while _js_truthy(self.eval(stmt["cond"], env)):
                try:
                    self.exec_stmt(stmt["body"], env)
                except _Break:
                    break
                except _Continue:
                    continue
        elif t == "Break":
            raise _Break()
        elif t == "Continue":
            raise _Continue()
        elif t == "ExprStmt":
            self.eval(stmt["expr"], env)
        elif t == "FunctionDecl":
            env.define(stmt["name"], stmt)
        else:
            raise NotImplementedError(f"JS stmt {t}")

    # ---- expressions ----
    def eval(self, node, env):
        t = node["type"]
        if t == "Literal":
            return node["value"]
        if t == "Identifier":
            if node["name"] == "Math":
                return "Math"
            return env.get(node["name"])
        if t == "ArrayLit":
            return [self.eval(e, env) for e in node["elems"]]
        if t == "Binary":
            return self._binary(node, env)
        if t == "Unary":
            v = self.eval(node["expr"], env)
            return (not _js_truthy(v)) if node["op"] == "!" else -v
        if t == "Ternary":
            return self.eval(node["then"] if _js_truthy(self.eval(node["cond"], env)) else node["else"], env)
        if t == "Index":
            obj = self.eval(node["obj"], env)
            idx = self.eval(node["index"], env)
            return obj[idx]
        if t == "Member":
            obj = self.eval(node["obj"], env)
            if node["prop"] == "length":
                return len(obj)
            raise AttributeError(f"JS: no property {node['prop']!r}")
        if t == "MethodCall":
            obj = self.eval(node["obj"], env)
            args = [self.eval(a, env) for a in node["args"]]
            m = node["method"]
            if obj == "Math":
                fn = {"floor": math.floor, "ceil": math.ceil, "abs": abs,
                      "max": max, "min": min}[m]
                return fn(*args)
            if m == "push":
                obj.append(args[0]); return len(obj)
            if m == "indexOf":
                try:
                    return obj.index(args[0])
                except ValueError:
                    return -1
            if m == "includes":
                return args[0] in obj
            if m == "slice":
                if len(args) == 0:
                    return list(obj)
                if len(args) > 1:
                    return obj[args[0]:args[1]]
                return obj[args[0]:]
            raise AttributeError(f"JS: no method {m!r}")
        if t == "Call":
            callee = node["callee"]
            args = [self.eval(a, env) for a in node["args"]]
            if callee["type"] == "Member" and callee["obj"]["type"] == "Identifier" \
                    and callee["obj"]["name"] == "Math":
                fn = {"floor": math.floor, "ceil": math.ceil, "abs": abs,
                      "max": max, "min": min}[callee["prop"]]
                return fn(*args)
            if callee["type"] == "Identifier":
                return self.call(callee["name"], args)
            raise NotImplementedError("JS: unsupported call")
        if t == "Assign":
            val = self.eval(node["value"], env)
            target = node["target"]
            if node["op"] != "=":
                cur = self._read_target(target, env)
                op = node["op"][0]
                val = {"+": cur + val, "-": cur - val, "*": cur * val,
                       "/": cur / val, "%": cur % val}[op]
            self._write_target(target, val, env)
            return val
        if t in ("PreInc", "PostInc"):
            cur = self._read_target(node["expr"], env)
            delta = 1 if node["op"] == "++" else -1
            self._write_target(node["expr"], cur + delta, env)
            return cur + delta if t == "PreInc" else cur
        raise NotImplementedError(f"JS expr {t}")

    def _binary(self, node, env):
        op = node["op"]
        if op == "&&":
            l = self.eval(node["left"], env)
            return self.eval(node["right"], env) if _js_truthy(l) else l
        if op == "||":
            l = self.eval(node["left"], env)
            return l if _js_truthy(l) else self.eval(node["right"], env)
        l = self.eval(node["left"], env)
        r = self.eval(node["right"], env)
        if op == "+":
            # JS: string concat wins if either side is a string
            if isinstance(l, str) or isinstance(r, str):
                return _js_str(l) + _js_str(r)
            return l + r
        if op == "-":
            return l - r
        if op == "*":
            return l * r
        if op == "/":
            return l / r if not (isinstance(l, int) and isinstance(r, int)) else l / r
        if op == "%":
            return l % r
        if op == "<":
            return l < r
        if op == ">":
            return l > r
        if op == "<=":
            return l <= r
        if op == ">=":
            return l >= r
        if op in ("==", "==="):
            return _js_eq(l, r)
        if op in ("!=", "!=="):
            return not _js_eq(l, r)
        raise NotImplementedError(f"JS op {op}")

    def _read_target(self, node, env):
        if node["type"] == "Identifier":
            return env.get(node["name"])
        if node["type"] == "Index":
            return self.eval(node["obj"], env)[self.eval(node["index"], env)]
        raise NotImplementedError(f"JS target {node['type']}")

    def _write_target(self, node, value, env):
        if node["type"] == "Identifier":
            env.set(node["name"], value)
            return
        if node["type"] == "Index":
            self.eval(node["obj"], env)[self.eval(node["index"], env)] = value
            return
        raise NotImplementedError(f"JS target {node['type']}")


def _js_str(v):
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def run_js_function(source, name, args):
    """Parse + interpret one JS function call. No node, no subprocess."""
    interp = JSInterpreter(parse_js(source))
    return interp.call(name, args)
