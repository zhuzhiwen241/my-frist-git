#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mini_lisp.py —— 一个 ~250 行的迷你 Lisp 解释器(纯标准库)

特性:
  * 词法作用域 + 闭包
  * 尾调用优化(TCO):while 循环代替递归,百万级尾递归不爆栈
  * 准引用(quasiquote ` , ,@)与宏(defmacro)
  * 带 行/列 号的错误定位
  * 常用内置函数
"""

import json
import operator
import re
from functools import reduce

# --------------------------------------------------------------------------- #
# 错误
# --------------------------------------------------------------------------- #


class LispError(Exception):
    def __init__(self, msg, line=None, col=None):
        self.msg, self.line, self.col = msg, line, col
        super().__init__(str(self))

    def __str__(self):
        if self.line is not None:
            return f"LispError [第 {self.line} 行,第 {self.col} 列]: {self.msg}"
        return f"LispError: {self.msg}"


# --------------------------------------------------------------------------- #
# 值类型
# --------------------------------------------------------------------------- #


class Symbol(str):
    """符号:用 str 子类与字符串字面量区分开。"""

    __slots__ = ()


class Procedure:
    def __init__(self, params, body, env, name="<lambda>"):
        self.params, self.body, self.env, self.name = params, body, env, name

    def __repr__(self):
        return f"#<procedure {self.name}>"


class Macro(Procedure):
    def __repr__(self):
        return f"#<macro {self.name}>"


# --------------------------------------------------------------------------- #
# 词法分析
# --------------------------------------------------------------------------- #

TOKEN_RE = re.compile(
    r"\s+"
    r"|;[^\n]*"
    r"|(?P<lparen>\()"
    r"|(?P<rparen>\))"
    r"|(?P<quote>')"
    r"|(?P<quasiquote>`)"
    r"|(?P<unquote_splice>,@)"
    r"|(?P<unquote>,)"
    r'|(?P<string>"(?:\\.|[^"\\])*")'
    r"|(?P<number>[+-]?\d+(?:\.\d+)?)"
    r"|(?P<symbol>[^\s()'`,\";]+)"
)


def pos_to_linecol(src, pos):
    line = src.count("\n", 0, pos) + 1
    col = pos - src.rfind("\n", 0, pos)
    return line, col


def tokenize(src):
    tokens, pos = [], 0
    while pos < len(src):
        m = TOKEN_RE.match(src, pos)
        if not m:
            line, col = pos_to_linecol(src, pos)
            raise LispError(f"无法识别的字符 {src[pos]!r}", line, col)
        if m.lastgroup not in ("ws", "comment") and m.lastgroup is not None:
            tokens.append((m.lastgroup, m.group(), m.start()))
        pos = m.end()
    return tokens


# --------------------------------------------------------------------------- #
# 语法分析(递归下降)
# --------------------------------------------------------------------------- #


class Parser:
    def __init__(self, tokens, src):
        self.tokens, self.i, self.src = tokens, 0, src

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def next(self):
        tok = self.peek()
        if tok is None:
            self.error("意外的输入结束")
        self.i += 1
        return tok

    def error(self, msg, tok=None):
        if tok is None:
            raise LispError(msg)
        line, col = pos_to_linecol(self.src, tok[2])
        raise LispError(msg, line, col)

    def parse(self):
        expr = self.parse_expr()
        if self.peek() is not None:
            self.error("多余的表达式", self.peek())
        return expr

    def parse_expr(self):
        kind, text, _ = tok = self.next()

        if kind == "number":
            return float(text) if "." in text else int(text)
        if kind == "string":
            return json.loads(text)          # 借 json 处理转义
        if kind == "symbol":
            return Symbol(text)

        if kind == "lparen":
            items = []
            while True:
                nxt = self.peek()
                if nxt is None:
                    self.error("括号未闭合")
                if nxt[0] == "rparen":
                    self.i += 1
                    return items
                items.append(self.parse_expr())

        if kind == "rparen":
            self.error("多余的 ')'", tok)

        # 语法糖:'x  `x  ,x  ,@x
        sugar = {
            "quote": "quote",
            "quasiquote": "quasiquote",
            "unquote": "unquote",
            "unquote_splice": "unquote-splicing",
        }
        return [Symbol(sugar[kind]), self.parse_expr()]


def parse(src):
    return Parser(tokenize(src), src).parse()


# --------------------------------------------------------------------------- #
# 环境
# --------------------------------------------------------------------------- #


class Env:
    def __init__(self, params=(), args=(), outer=None):
        self.vars = dict(zip(params, args))
        self.outer = outer

    def find(self, name):
        env = self
        while env is not None:
            if name in env.vars:
                return env.vars[name]
            env = env.outer
        raise LispError(f"未绑定变量: {name}")

    def set(self, name, value):
        env = self
        while env is not None:
            if name in env.vars:
                env.vars[name] = value
                return
            env = env.outer
        raise LispError(f"set! 未绑定变量: {name}")


# --------------------------------------------------------------------------- #
# 求值器(带 TCO)
# --------------------------------------------------------------------------- #


def truthy(x):
    return x is not False and x != []


def quasiquote(expr, env, depth=1):
    if not isinstance(expr, list):
        return expr
    if expr and isinstance(expr[0], Symbol):
        head = expr[0]
        if head == "unquote":
            if depth == 1:
                return evaluate(expr[1], env)
            return [head] + [quasiquote(e, env, depth - 1) for e in expr[1:]]
        if head == "quasiquote":
            return [head] + [quasiquote(e, env, depth + 1) for e in expr[1:]]

    out = []
    for item in expr:
        if (
            depth == 1
            and isinstance(item, list)
            and item
            and isinstance(item[0], Symbol)
            and item[0] == "unquote-splicing"
        ):
            out.extend(evaluate(item[1], env))
        else:
            out.append(quasiquote(item, env, depth))
    return out


def evaluate(expr, env):
    while True:
        # ---- 原子 ----
        if isinstance(expr, Symbol):
            return env.find(expr)
        if not isinstance(expr, list):
            return expr
        if not expr:
            return []

        op = expr[0]

        # ---- 特殊形式 ----
        if isinstance(op, Symbol):
            if op == "quote":
                return expr[1]

            if op == "quasiquote":
                return quasiquote(expr[1], env)

            if op == "if":
                if len(expr) != 4:
                    raise LispError("if 需要 3 个参数: (if test conseq alt)")
                expr = expr[2] if truthy(evaluate(expr[1], env)) else expr[3]
                continue                                  # ← 尾位置

            if op == "define":
                target = expr[1]
                if isinstance(target, list):              # (define (f x) body)
                    env.vars[target[0]] = Procedure(target[1:], expr[2:], env, target[0])
                else:                                     # (define x val)
                    env.vars[target] = evaluate(expr[2], env)
                return None

            if op == "set!":
                env.set(expr[1], evaluate(expr[2], env))
                return None

            if op == "lambda":
                return Procedure(expr[1], expr[2:], env)

            if op == "begin":
                if len(expr) == 1:
                    return None
                for e in expr[1:-1]:
                    evaluate(e, env)
                expr = expr[-1]
                continue                                  # ← 尾位置

            if op == "let":                               # 转成 lambda 调用
                names = [b[0] for b in expr[1]]
                vals = [b[1] for b in expr[1]]
                expr = [[Symbol("lambda"), names] + expr[2:]] + vals
                continue

            if op == "and":
                val = True
                for e in expr[1:]:
                    val = evaluate(e, env)
                    if not truthy(val):
                        return val
                return val

            if op == "or":
                for e in expr[1:]:
                    val = evaluate(e, env)
                    if truthy(val):
                        return val
                return False

            if op == "defmacro":                          # (defmacro name (args) body)
                env.vars[expr[1]] = Macro(expr[2], expr[3:], env, expr[1])
                return None

        # ---- 函数调用 ----
        fn = evaluate(op, env)

        if isinstance(fn, Macro):                         # 宏:先展开再继续求值
            macro_env = Env(fn.params, expr[1:], fn.env)
            for e in fn.body[:-1]:
                evaluate(e, macro_env)
            expr = evaluate(fn.body[-1], macro_env)
            continue

        args = [evaluate(a, env) for a in expr[1:]]

        if isinstance(fn, Procedure):                     # ← 尾调用优化核心
            env = Env(fn.params, args, fn.env)
            if not fn.body:
                return None
            for e in fn.body[:-1]:
                evaluate(e, env)
            expr = fn.body[-1]
            continue

        if callable(fn):
            return fn(*args)

        raise LispError(f"{fn!r} 不是可调用的")


# --------------------------------------------------------------------------- #
# 输出格式化
# --------------------------------------------------------------------------- #


def to_str(x):
    if x is True:
        return "#t"
    if x is False:
        return "#f"
    if x is None:
        return "nil"
    if isinstance(x, Symbol):
        return str(x)
    if isinstance(x, str):
        return json.dumps(x, ensure_ascii=False)
    if isinstance(x, list):
        return "(" + " ".join(to_str(i) for i in x) + ")"
    if isinstance(x, (Procedure, Macro)):
        return repr(x)
    return str(x)


# --------------------------------------------------------------------------- #
# 标准库
# --------------------------------------------------------------------------- #


def _sub(a, *rest):
    return -a if not rest else reduce(operator.sub, rest, a)


def _div(a, *rest):
    r = reduce(operator.truediv, rest, a)
    return int(r) if isinstance(r, float) and r.is_integer() else r


def standard_env():
    env = Env()
    env.vars.update(
        {
            "+": lambda *a: sum(a),
            "-": _sub,
            "*": lambda *a: reduce(operator.mul, a, 1),
            "/": _div,
            "=": operator.eq,
            "<": operator.lt,
            ">": operator.gt,
            "<=": operator.le,
            ">=": operator.ge,
            "not": lambda x: not truthy(x),
            "list": lambda *a: list(a),
            "car": lambda x: x[0],
            "cdr": lambda x: x[1:],
            "cons": lambda a, d: [a] + d,
            "null?": lambda x: x == [],
            "length": len,
            "append": lambda *xs: [i for x in xs for i in x],
            "reverse": lambda x: list(reversed(x)),
            "print": lambda *a: print(*[to_str(i) for i in a]),
            "nil": [],
            "true": True,
            "false": False,
        }
    )
    return env


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


def run(src, env=None):
    env = env if env is not None else standard_env()
    return evaluate(parse(src), env), env


DEMO = r"""
; ---------- 基础 ----------
(+ 1 2 3 4)
(define x 42)
(* x x)

; ---------- 递归 ----------
(define (fib n)
  (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))
(fib 20)

; ---------- 闭包 + 可变状态 ----------
(define (make-counter)
  (let ((n 0))
    (lambda ()
      (set! n (+ n 1))
      n)))
(define c (make-counter))
(list (c) (c) (c))

; ---------- 尾调用优化:十万次循环不爆栈 ----------
(define (loop n acc)
  (if (= n 0) acc (loop (- n 1) (+ acc 1))))
(loop 100000 0)

; ---------- 宏 ----------
(defmacro unless (test body)
  `(if ,test () ,body))
(unless (= 1 2) 'ok)

; ---------- 准引用拼接 ----------
(define xs '(2 3 4))
`(1 ,@xs 5)
"""

if __name__ == "__main__":
    env = standard_env()
    for form in re.split(r"\n\s*\n", DEMO.strip()):
        for line in form.splitlines():
            if line.strip().startswith(";"):
                print(f"\033[36m{line.strip()}\033[0m")
            elif line.strip():
                try:
                    result, env = run(line, env)
                    print(f"  => {to_str(result)}")
                except LispError as e:
                    print(f"  !! {e}")                   