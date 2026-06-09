"""Compile common PySpark-style lambdas into native C++ transform ops."""

from __future__ import annotations

import ast
import dis
import inspect
import textwrap
from dataclasses import dataclass
from types import CodeType
from typing import Any, Callable, Optional

from pyvega._protocol import FloatCompareOp, NativeOp, StageOp


NativeKind = str  # "int64", "float64", "str"


@dataclass(frozen=True)
class NativeStage:
    op: StageOp
    native_op: NativeOp
    params: tuple[int | float | str, ...]
    num_partitions: int = 0


def _closure_ints(func: Callable[..., Any]) -> dict[str, int]:
    code = func.__code__
    if not func.__closure__ or not code.co_freevars:
        return {}
    out: dict[str, int] = {}
    for name, cell in zip(code.co_freevars, func.__closure__):
        value = cell.cell_contents
        if isinstance(value, int) and not isinstance(value, bool):
            out[name] = int(value)
    return out


def _int_env(func: Callable[..., Any]) -> dict[str, int]:
    """Integer names visible to a lambda (closure cells and module globals)."""
    env = _closure_ints(func)
    for name, value in func.__globals__.items():
        if isinstance(value, int) and not isinstance(value, bool):
            env.setdefault(name, int(value))
    return env


def _closure_floats(func: Callable[..., Any]) -> dict[str, float]:
    code = func.__code__
    if not func.__closure__ or not code.co_freevars:
        return {}
    out: dict[str, float] = {}
    for name, cell in zip(code.co_freevars, func.__closure__):
        value = cell.cell_contents
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[name] = float(value)
    return out


def _float_env(func: Callable[..., Any]) -> dict[str, float]:
    env = _closure_floats(func)
    for name, value in func.__globals__.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            env.setdefault(name, float(value))
    return env


def _closure_strs(func: Callable[..., Any]) -> dict[str, str]:
    code = func.__code__
    if not func.__closure__ or not code.co_freevars:
        return {}
    out: dict[str, str] = {}
    for name, cell in zip(code.co_freevars, func.__closure__):
        value = cell.cell_contents
        if isinstance(value, str):
            out[name] = value
    return out


def _str_env(func: Callable[..., Any]) -> dict[str, str]:
    env = _closure_strs(func)
    for name, value in func.__globals__.items():
        if isinstance(value, str):
            env.setdefault(name, value)
    return env


def _const_float(node: ast.AST, env: dict[str, float] | None = None) -> Optional[float]:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
    if env and isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    return None


def _const_str(node: ast.AST, env: dict[str, str] | None = None) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if env and isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    return None


def _lambda_signature(func: Callable[..., Any]) -> tuple[int, tuple[str, ...]]:
    code = func.__code__
    return code.co_argcount, tuple(code.co_varnames[: code.co_argcount])


def _lambda_matches(lam: ast.Lambda, argcount: int, argnames: tuple[str, ...]) -> bool:
    if len(lam.args.args) != argcount:
        return False
    return tuple(arg.arg for arg in lam.args.args) == argnames


def _parse_source_tree(text: str) -> ast.AST | None:
    stripped = text.strip()
    candidates = [stripped]
    if stripped.startswith("."):
        candidates.append(f"(ROOT{stripped})")
    for candidate in candidates:
        for mode in ("eval", "exec"):
            try:
                return ast.parse(candidate, mode=mode)
            except SyntaxError:
                continue
    return None


def _const(node: ast.AST, env: dict[str, int] | None = None) -> Optional[int]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if env and isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    return None


def _name(node: ast.AST) -> Optional[str]:
    return node.id if isinstance(node, ast.Name) else None


def _affine_mod(node: ast.AST, var: str, env: dict[str, int] | None = None) -> Optional[tuple[int, int, int]]:
    """Match (var * mul + add) % mod."""
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mod):
        return None
    inner = node.left
    mod = _const(node.right, env)
    if mod is None or mod <= 0:
        return None

    mul, add = 1, 0
    if isinstance(inner, ast.Name) and inner.id == var:
        return mul, add, mod
    if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Add):
        if isinstance(inner.left, ast.BinOp) and isinstance(inner.left.op, ast.Mult):
            if _name(inner.left.left) == var:
                mul = _const(inner.left.right, env)
            elif _name(inner.left.right) == var:
                mul = _const(inner.left.left, env)
            else:
                return None
            add = _const(inner.right, env)
        elif _name(inner.left) == var:
            add = _const(inner.right, env)
        elif _name(inner.right) == var:
            add = _const(inner.left, env)
        else:
            return None
        if mul is None or add is None:
            return None
        return mul, add, mod
    if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Mult):
        if _name(inner.left) == var:
            mul = _const(inner.right, env)
        elif _name(inner.right) == var:
            mul = _const(inner.left, env)
        else:
            return None
        if mul is None:
            return None
        return mul, add, mod
    return None


def _affine_f64(node: ast.AST, var: str, env: dict[str, float] | None = None) -> Optional[tuple[float, float]]:
    """Match var * mul + add or var * mul or var + add."""
    mul, add = 1.0, 0.0
    if isinstance(node, ast.Name) and node.id == var:
        return mul, add
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        if _name(node.left) == var:
            m = _const_float(node.right, env)
        elif _name(node.right) == var:
            m = _const_float(node.left, env)
        else:
            return None
        if m is None:
            return None
        return m, add
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        if _name(node.left) == var:
            a = _const_float(node.right, env)
            if a is None:
                return None
            return mul, a
        if _name(node.right) == var:
            a = _const_float(node.left, env)
            if a is None:
                return None
            return mul, a
        if (
            isinstance(node.left, ast.BinOp)
            and isinstance(node.left.op, ast.Mult)
            and _name(node.left.left) == var
        ):
            m = _const_float(node.left.right, env)
            a = _const_float(node.right, env)
            if m is None or a is None:
                return None
            return m, a
        if (
            isinstance(node.left, ast.BinOp)
            and isinstance(node.left.op, ast.Mult)
            and _name(node.left.right) == var
        ):
            m = _const_float(node.left.left, env)
            a = _const_float(node.right, env)
            if m is None or a is None:
                return None
            return m, a
    return None


def _compare_op(node: ast.cmpop) -> Optional[FloatCompareOp]:
    if isinstance(node, ast.Eq):
        return FloatCompareOp.EQ
    if isinstance(node, ast.Gt):
        return FloatCompareOp.GT
    if isinstance(node, ast.Lt):
        return FloatCompareOp.LT
    if isinstance(node, ast.GtE):
        return FloatCompareOp.GE
    if isinstance(node, ast.LtE):
        return FloatCompareOp.LE
    return None


def _lambda_to_funcdef(lam: ast.Lambda) -> ast.FunctionDef:
    return ast.FunctionDef(
        name="_lambda",
        args=lam.args,
        body=[ast.Return(value=lam.body, lineno=lam.lineno, col_offset=lam.col_offset)],
        decorator_list=[],
        lineno=lam.lineno,
        col_offset=lam.col_offset,
    )


def _lambda_ast(func: Callable[..., Any], call_name: str | None = None) -> ast.FunctionDef:
    argcount, argnames = _lambda_signature(func)
    code = func.__code__
    target_lineno = code.co_firstlineno
    target_col = getattr(code, "co_coloffset", None)

    def pick_lambda(lam: ast.Lambda) -> ast.FunctionDef:
        return _lambda_to_funcdef(lam)

    def find_by_call_name(tree: ast.AST) -> ast.Lambda | None:
        if call_name is None:
            return None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_node = node.func
            attr = None
            if isinstance(func_node, ast.Attribute):
                attr = func_node.attr
            elif isinstance(func_node, ast.Name):
                attr = func_node.id
            if attr not in {call_name, "flat_map", "reduceByKey", "reduce_by_key"}:
                continue
            if call_name == "flat_map" and attr not in {"flat_map", "flatMap"}:
                continue
            if call_name == "map" and attr != "map":
                continue
            if call_name == "filter" and attr != "filter":
                continue
            if call_name == "reduce_by_key" and attr not in {
                "reduce_by_key",
                "reduceByKey",
            }:
                continue
            if not node.args:
                continue
            lam = node.args[0]
            if isinstance(lam, ast.Lambda) and _lambda_matches(lam, argcount, argnames):
                return lam
        return None

    try:
        source = textwrap.dedent(inspect.getsource(func).strip())
        if source.startswith("lambda"):
            lam = ast.parse(source, mode="eval").body
            if isinstance(lam, ast.Lambda) and _lambda_matches(lam, argcount, argnames):
                return pick_lambda(lam)
        tree = ast.parse(source)
        node = tree.body[0]
        if isinstance(node, ast.FunctionDef):
            return node
    except (OSError, SyntaxError, ValueError):
        pass

    lines, start_line = inspect.getsourcelines(func)
    text = textwrap.dedent("".join(lines)).strip()
    tree = _parse_source_tree(text)
    if tree is not None:
        by_call = find_by_call_name(tree)
        if by_call is not None:
            return pick_lambda(by_call)

        matches: list[ast.Lambda] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Lambda) and _lambda_matches(node, argcount, argnames):
                matches.append(node)
        if len(matches) == 1:
            return pick_lambda(matches[0])
        for node in matches:
            abs_line = start_line + node.lineno - 1
            if abs_line != target_lineno:
                continue
            if target_col is not None and node.col_offset != target_col:
                continue
            return pick_lambda(node)
        for node in matches:
            abs_line = start_line + node.lineno - 1
            if abs_line == target_lineno:
                return pick_lambda(node)
    raise ValueError("could not locate lambda source")


def compile_map(func: Callable[..., Any], native_kind: NativeKind = "int64") -> Optional[NativeStage]:
    if native_kind == "float64":
        return compile_map_f64(func)
    if native_kind == "str":
        return compile_map_str(func)
    env = _int_env(func)
    tree = _lambda_ast(func, "map")
    if not isinstance(tree, ast.FunctionDef) or len(tree.args.args) != 1:
        return None
    var = tree.args.args[0].arg
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Return):
        return None
    expr = tree.body[0].value
    if isinstance(expr, ast.Tuple) and len(expr.elts) == 2:
        if _name(expr.elts[0]) == var:
            vconst = _const(expr.elts[1], env)
            if vconst is not None:
                return NativeStage(
                    StageOp.MAP,
                    NativeOp.MAP_TO_PAIR_CONST,
                    (vconst,),
                )
        if (
            isinstance(expr.elts[0], ast.BinOp)
            and isinstance(expr.elts[0].op, ast.Mod)
            and _name(expr.elts[0].left) == var
        ):
            kmod = _const(expr.elts[0].right, env)
            vconst = _const(expr.elts[1], env)
            if kmod is not None and vconst is not None:
                return NativeStage(
                    StageOp.MAP,
                    NativeOp.MAP_TO_PAIR_MOD_CONST,
                    (kmod, vconst),
                )
        k = _affine_mod(expr.elts[0], var, env)
        v = _affine_mod(expr.elts[1], var, env)
        if k and v:
            return NativeStage(
                StageOp.MAP,
                NativeOp.MAP_TO_PAIR_AFFINE,
                k + v,
            )
    aff = _affine_mod(expr, var, env)
    if aff:
        return NativeStage(StageOp.MAP, NativeOp.MAP_AFFINE, aff)
    return None


def compile_map_f64(func: Callable[..., Any]) -> Optional[NativeStage]:
    env = _float_env(func)
    tree = _lambda_ast(func, "map")
    if not isinstance(tree, ast.FunctionDef) or len(tree.args.args) != 1:
        return None
    var = tree.args.args[0].arg
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Return):
        return None
    expr = tree.body[0].value
    if isinstance(expr, ast.Tuple) and len(expr.elts) == 2:
        vconst = _const(expr.elts[1], {k: int(v) for k, v in env.items() if v == int(v)})
        if vconst is None:
            v = _const_float(expr.elts[1], env)
            if v is not None and v == int(v):
                vconst = int(v)
        if vconst is not None:
            left = expr.elts[0]
            if _name(left) == var:
                return NativeStage(StageOp.MAP, NativeOp.MAP_TO_PAIR_CONST_F64, (vconst,))
            if (
                isinstance(left, ast.BinOp)
                and isinstance(left.op, ast.Mod)
                and isinstance(left.left, ast.Call)
                and isinstance(left.left.func, ast.Name)
                and left.left.func.id == "int"
                and len(left.left.args) == 1
                and _name(left.left.args[0]) == var
            ):
                kmod = _const(left.right, {k: int(v) for k, v in env.items() if v == int(v)})
                if kmod is None:
                    kmod_f = _const_float(left.right, env)
                    if kmod_f is not None and kmod_f == int(kmod_f):
                        kmod = int(kmod_f)
                if kmod is not None:
                    return NativeStage(
                        StageOp.MAP,
                        NativeOp.MAP_TO_PAIR_MOD_CONST_F64,
                        (kmod, vconst),
                    )
    aff = _affine_f64(expr, var, env)
    if aff:
        return NativeStage(StageOp.MAP, NativeOp.MAP_AFFINE_F64, aff)
    return None


def compile_map_str(func: Callable[..., Any]) -> Optional[NativeStage]:
    env = _int_env(func)
    tree = _lambda_ast(func, "map")
    if not isinstance(tree, ast.FunctionDef) or len(tree.args.args) != 1:
        return None
    var = tree.args.args[0].arg
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Return):
        return None
    expr = tree.body[0].value
    if isinstance(expr, ast.Tuple) and len(expr.elts) == 2 and _name(expr.elts[0]) == var:
        vconst = _const(expr.elts[1], env)
        if vconst is not None:
            return NativeStage(StageOp.MAP, NativeOp.MAP_TO_PAIR_CONST_STR, (vconst,))
    return None


def compile_filter(func: Callable[..., Any], native_kind: NativeKind = "int64") -> Optional[NativeStage]:
    if native_kind == "float64":
        return compile_filter_f64(func)
    if native_kind == "str":
        return compile_filter_str(func)
    env = _int_env(func)
    tree = _lambda_ast(func, "filter")
    if not isinstance(tree, ast.FunctionDef) or len(tree.args.args) != 1:
        return None
    var = tree.args.args[0].arg
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Return):
        return None
    expr = tree.body[0].value
    if not isinstance(expr, ast.Compare) or len(expr.ops) != 1 or len(expr.comparators) != 1:
        return None
    if not isinstance(expr.ops[0], ast.Eq):
        return None
    eq = _const(expr.comparators[0], env)
    if eq is None:
        return None
    if (
        isinstance(expr.left, ast.BinOp)
        and isinstance(expr.left.op, ast.Mod)
        and _name(expr.left.left) == var
    ):
        mod = _const(expr.left.right, env)
        if mod is not None and mod > 0:
            return NativeStage(StageOp.FILTER, NativeOp.FILTER_MOD_EQ, (mod, eq))
    return None


def compile_filter_f64(func: Callable[..., Any]) -> Optional[NativeStage]:
    env = _float_env(func)
    tree = _lambda_ast(func, "filter")
    if not isinstance(tree, ast.FunctionDef) or len(tree.args.args) != 1:
        return None
    var = tree.args.args[0].arg
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Return):
        return None
    expr = tree.body[0].value
    if not isinstance(expr, ast.Compare) or len(expr.ops) != 1 or len(expr.comparators) != 1:
        return None
    if _name(expr.left) != var:
        return None
    cmp_op = _compare_op(expr.ops[0])
    value = _const_float(expr.comparators[0], env)
    if cmp_op is None or value is None:
        return None
    return NativeStage(
        StageOp.FILTER,
        NativeOp.FILTER_COMPARE_F64,
        (int(cmp_op), value),
    )


def compile_filter_str(func: Callable[..., Any]) -> Optional[NativeStage]:
    env = _int_env(func)
    tree = _lambda_ast(func, "filter")
    if not isinstance(tree, ast.FunctionDef) or len(tree.args.args) != 1:
        return None
    var = tree.args.args[0].arg
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Return):
        return None
    expr = tree.body[0].value
    if not isinstance(expr, ast.Compare) or len(expr.ops) != 1 or len(expr.comparators) != 1:
        return None
    if not isinstance(expr.ops[0], ast.Gt):
        return None
    if (
        isinstance(expr.left, ast.Call)
        and isinstance(expr.left.func, ast.Name)
        and expr.left.func.id == "len"
        and len(expr.left.args) == 1
        and _name(expr.left.args[0]) == var
    ):
        min_len = _const(expr.comparators[0], env)
        if min_len is not None:
            return NativeStage(StageOp.FILTER, NativeOp.FILTER_LENGTH_GT, (min_len,))
    return None


def compile_flat_map(func: Callable[..., Any], native_kind: NativeKind = "int64") -> Optional[NativeStage]:
    if native_kind == "float64":
        return compile_flat_map_f64(func)
    if native_kind == "str":
        return compile_flat_map_str(func)
    env = _int_env(func)
    tree = _lambda_ast(func, "flat_map")
    if not isinstance(tree, ast.FunctionDef) or len(tree.args.args) != 1:
        return None
    var = tree.args.args[0].arg
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Return):
        return None
    expr = tree.body[0].value
    if not isinstance(expr, ast.Tuple) or len(expr.elts) < 1:
        return None
    offsets: list[int] = []
    for elt in expr.elts:
        if _name(elt) == var:
            offsets.append(0)
        elif (
            isinstance(elt, ast.BinOp)
            and isinstance(elt.op, ast.Add)
            and _name(elt.left) == var
        ):
            off = _const(elt.right, env)
            if off is None:
                return None
            offsets.append(off)
        elif (
            isinstance(elt, ast.BinOp)
            and isinstance(elt.op, ast.Add)
            and _name(elt.right) == var
        ):
            off = _const(elt.left, env)
            if off is None:
                return None
            offsets.append(off)
        else:
            return None
    if offsets != list(range(len(offsets))):
        return None
    return NativeStage(
        StageOp.FLAT_MAP,
        NativeOp.FLAT_MAP_RANGE,
        (0, len(offsets)),
    )


def _const_offset(node: ast.AST, env: dict[str, float] | None = None) -> Optional[int]:
    """Integer offset from a numeric literal or name (e.g. 1.0 -> 1)."""
    value = _const_float(node, env)
    if value is None or value != int(value):
        return None
    return int(value)


def compile_flat_map_f64(func: Callable[..., Any]) -> Optional[NativeStage]:
    env = _float_env(func)
    tree = _lambda_ast(func, "flat_map")
    if not isinstance(tree, ast.FunctionDef) or len(tree.args.args) != 1:
        return None
    var = tree.args.args[0].arg
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Return):
        return None
    expr = tree.body[0].value
    if not isinstance(expr, ast.Tuple) or len(expr.elts) < 1:
        return None
    offsets: list[int] = []
    for elt in expr.elts:
        if _name(elt) == var:
            offsets.append(0)
        elif (
            isinstance(elt, ast.BinOp)
            and isinstance(elt.op, ast.Add)
            and _name(elt.left) == var
        ):
            off = _const_offset(elt.right, env)
            if off is None:
                return None
            offsets.append(off)
        elif (
            isinstance(elt, ast.BinOp)
            and isinstance(elt.op, ast.Add)
            and _name(elt.right) == var
        ):
            off = _const_offset(elt.left, env)
            if off is None:
                return None
            offsets.append(off)
        else:
            return None
    if offsets != list(range(len(offsets))):
        return None
    return NativeStage(
        StageOp.FLAT_MAP,
        NativeOp.FLAT_MAP_RANGE_F64,
        (0, len(offsets)),
    )


def compile_flat_map_str(func: Callable[..., Any]) -> Optional[NativeStage]:
    str_env = _str_env(func)
    tree = _lambda_ast(func, "flat_map")
    if not isinstance(tree, ast.FunctionDef) or len(tree.args.args) != 1:
        return None
    var = tree.args.args[0].arg
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Return):
        return None
    expr = tree.body[0].value
    if not isinstance(expr, ast.Call):
        return None
    if not isinstance(expr.func, ast.Attribute) or expr.func.attr != "split":
        return None
    if len(expr.args) >= 1 and _name(expr.args[0]) != var:
        return None
    if expr.keywords:
        return None
    if not expr.args:
        return NativeStage(StageOp.FLAT_MAP, NativeOp.FLAT_MAP_SPLIT, ("",))
    delim = _const_str(expr.args[0], str_env)
    if delim is None:
        return None
    return NativeStage(StageOp.FLAT_MAP, NativeOp.FLAT_MAP_SPLIT, (delim,))


def compile_reduce_by_key(func: Callable[..., Any]) -> Optional[NativeStage]:
    tree = _lambda_ast(func, "reduce_by_key")
    if not isinstance(tree, ast.FunctionDef) or len(tree.args.args) != 2:
        return None
    a_name = tree.args.args[0].arg
    b_name = tree.args.args[1].arg
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Return):
        return None
    expr = tree.body[0].value
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        left, right = _name(expr.left), _name(expr.right)
        if (left == a_name and right == b_name) or (left == b_name and right == a_name):
            return NativeStage(StageOp.REDUCE_BY_KEY, NativeOp.REDUCE_BY_KEY_SUM, ())
    return None


_BINOP_ARG = {"+": 0, "*": 5, "%": 6}
_COMPARE_NAME = {
    "==": FloatCompareOp.EQ,
    ">": FloatCompareOp.GT,
    "<": FloatCompareOp.LT,
    ">=": FloatCompareOp.GE,
    "<=": FloatCompareOp.LE,
}


def _bytecode_instructions(func: Callable[..., Any]) -> list[dis.Instruction]:
    return [
        ins
        for ins in dis.get_instructions(func)
        if ins.opname not in ("RESUME", "RETURN_VALUE", "CACHE")
    ]


def _is_binop(ins: dis.Instruction, op: str) -> bool:
    if ins.opname == "BINARY_OP":
        return ins.arg == _BINOP_ARG[op]
    legacy = {"+": "BINARY_ADD", "*": "BINARY_MULTIPLY", "%": "BINARY_MODULO"}
    return ins.opname == legacy[op]


def _const_value(ins: dis.Instruction) -> int | float | str | None:
    if ins.opname != "LOAD_CONST":
        return None
    value = ins.argval
    if isinstance(value, bool):
        return None
    return value


def _int_const(ins: dis.Instruction, env: dict[str, int] | None = None) -> Optional[int]:
    value = _const_value(ins)
    if isinstance(value, int):
        return value
    if env and isinstance(value, str) and value in env:
        return env[value]
    return None


def _float_const(ins: dis.Instruction, env: dict[str, float] | None = None) -> Optional[float]:
    value = _const_value(ins)
    if isinstance(value, (int, float)):
        return float(value)
    if env and isinstance(value, str) and value in env:
        return env[value]
    return None


def _match_affine_mod_bytecode(
    ins: list[dis.Instruction],
    env: dict[str, int] | None = None,
) -> Optional[tuple[int, int, int]]:
    """Match x * mul + add % mod."""
    if len(ins) != 7:
        return None
    if ins[0].opname not in ("LOAD_FAST", "LOAD_FAST_LOAD_FAST"):
        return None
    mul = _int_const(ins[1], env)
    if mul is None or not _is_binop(ins[2], "*"):
        return None
    add = _int_const(ins[3], env)
    if add is None or not _is_binop(ins[4], "+"):
        return None
    mod = _int_const(ins[5], env)
    if mod is None or mod <= 0 or not _is_binop(ins[6], "%"):
        return None
    return mul, add, mod


def _match_mod_bytecode(
    ins: list[dis.Instruction],
    env: dict[str, int] | None = None,
) -> Optional[int]:
    if len(ins) != 3:
        return None
    if ins[0].opname != "LOAD_FAST":
        return None
    mod = _int_const(ins[1], env)
    if mod is None or mod <= 0 or not _is_binop(ins[2], "%"):
        return None
    return mod


def _match_flat_map_range_bytecode(
    ins: list[dis.Instruction],
    env: dict[str, int] | None = None,
) -> Optional[tuple[int, int]]:
    if len(ins) < 2 or ins[-1].opname != "BUILD_TUPLE":
        return None
    count = ins[-1].arg
    if count is None or count < 1:
        return None
    body = ins[:-1]
    i = 0
    if body[i].opname not in ("LOAD_FAST", "LOAD_FAST_LOAD_FAST"):
        return None
    used_load_fast_load_fast = body[i].opname == "LOAD_FAST_LOAD_FAST"
    i += 1
    for offset in range(1, count):
        if used_load_fast_load_fast and offset == 1:
            if i + 1 >= len(body):
                return None
            off = _int_const(body[i], env)
            if off is None or off != offset or not _is_binop(body[i + 1], "+"):
                return None
            i += 2
            continue
        if i + 2 >= len(body):
            return None
        if body[i].opname != "LOAD_FAST":
            return None
        off = _int_const(body[i + 1], env)
        if off is None or off != offset or not _is_binop(body[i + 2], "+"):
            return None
        i += 3
    if i != len(body):
        return None
    return 0, count


def _match_flat_map_range_f64_bytecode(
    ins: list[dis.Instruction],
    env: dict[str, float] | None = None,
) -> Optional[tuple[int, int]]:
    if len(ins) < 2 or ins[-1].opname != "BUILD_TUPLE":
        return None
    count = ins[-1].arg
    if count is None or count < 1:
        return None
    body = ins[:-1]
    i = 0
    if body[i].opname not in ("LOAD_FAST", "LOAD_FAST_LOAD_FAST"):
        return None
    used_load_fast_load_fast = body[i].opname == "LOAD_FAST_LOAD_FAST"
    i += 1
    for offset in range(1, count):
        if used_load_fast_load_fast and offset == 1:
            if i + 1 >= len(body):
                return None
            off_f = _float_const(body[i], env)
            if off_f is None or off_f != float(offset) or not _is_binop(body[i + 1], "+"):
                return None
            i += 2
            continue
        if i + 2 >= len(body):
            return None
        if body[i].opname != "LOAD_FAST":
            return None
        off_f = _float_const(body[i + 1], env)
        if off_f is None or off_f != float(offset) or not _is_binop(body[i + 2], "+"):
            return None
        i += 3
    if i != len(body):
        return None
    return 0, count


def compile_map_bytecode(
    func: Callable[..., Any],
    native_kind: NativeKind = "int64",
) -> Optional[NativeStage]:
    code: CodeType = func.__code__
    if code.co_argcount != 1:
        return None
    ins = _bytecode_instructions(func)
    if native_kind == "float64":
        env = _float_env(func)
        if len(ins) == 5 and ins[0].opname == "LOAD_FAST":
            mul = _float_const(ins[1], env)
            if mul is not None and _is_binop(ins[2], "*"):
                add = _float_const(ins[3], env)
                if add is not None and _is_binop(ins[4], "+"):
                    return NativeStage(StageOp.MAP, NativeOp.MAP_AFFINE_F64, (mul, add))
        if (
            len(ins) == 7
            and ins[0].opname == "LOAD_GLOBAL"
            and ins[0].argval == int
            and ins[1].opname == "LOAD_FAST"
            and ins[2].opname == "CALL"
        ):
            int_env = {k: int(v) for k, v in env.items() if v == int(v)}
            kmod = _int_const(ins[3], int_env)
            if kmod is not None and _is_binop(ins[4], "%"):
                vconst = _int_const(ins[5], int_env)
                if vconst is not None and ins[6].opname == "BUILD_TUPLE":
                    return NativeStage(
                        StageOp.MAP,
                        NativeOp.MAP_TO_PAIR_MOD_CONST_F64,
                        (kmod, vconst),
                    )
        return None
    if native_kind == "str":
        env = _int_env(func)
        if (
            len(ins) == 3
            and ins[0].opname == "LOAD_FAST"
            and ins[1].opname == "LOAD_CONST"
        ):
            vconst = _int_const(ins[1], env)
            if vconst is not None and ins[2].opname == "BUILD_TUPLE":
                return NativeStage(StageOp.MAP, NativeOp.MAP_TO_PAIR_CONST_STR, (vconst,))
        return None

    env = _int_env(func)
    aff = _match_affine_mod_bytecode(ins, env)
    if aff:
        return NativeStage(StageOp.MAP, NativeOp.MAP_AFFINE, aff)
    if len(ins) == 5 and ins[0].opname == "LOAD_FAST" and ins[3].opname == "LOAD_FAST":
        if ins[1].opname == "LOAD_CONST" and ins[4].opname == "BUILD_TUPLE":
            if ins[2].opname == "BINARY_OP" and ins[2].argrepr == "%":
                kmod = _int_const(ins[1], env)
                if kmod is not None:
                    return NativeStage(
                        StageOp.MAP,
                        NativeOp.MAP_TO_PAIR_MOD_IDENTITY,
                        (kmod,),
                    )
    if len(ins) == 5 and ins[0].opname == "LOAD_FAST" and ins[3].opname == "LOAD_CONST":
        kmod = _match_mod_bytecode(ins[:3], env)
        vconst = _int_const(ins[3], env)
        if kmod is not None and vconst is not None and ins[4].opname == "BUILD_TUPLE":
            return NativeStage(StageOp.MAP, NativeOp.MAP_TO_PAIR_MOD_CONST, (kmod, vconst))
    if len(ins) == 3 and ins[0].opname == "LOAD_FAST" and ins[2].opname == "BUILD_TUPLE":
        vconst = _int_const(ins[1], env)
        if vconst is not None:
            return NativeStage(StageOp.MAP, NativeOp.MAP_TO_PAIR_CONST, (vconst,))
    return None


def compile_filter_bytecode(
    func: Callable[..., Any],
    native_kind: NativeKind = "int64",
) -> Optional[NativeStage]:
    code: CodeType = func.__code__
    if code.co_argcount != 1:
        return None
    ins = _bytecode_instructions(func)
    if native_kind == "float64":
        env = _float_env(func)
        if len(ins) != 3 or ins[0].opname != "LOAD_FAST" or ins[1].opname != "LOAD_CONST":
            return None
        if ins[2].opname != "COMPARE_OP":
            return None
        cmp_op = _COMPARE_NAME.get(ins[2].argrepr)
        value = _float_const(ins[1], env)
        if cmp_op is None or value is None:
            return None
        return NativeStage(
            StageOp.FILTER,
            NativeOp.FILTER_COMPARE_F64,
            (int(cmp_op), value),
        )
    if native_kind == "str":
        env = _int_env(func)
        if (
            len(ins) == 6
            and ins[0].opname == "LOAD_GLOBAL"
            and ins[0].argval == len
            and ins[2].opname == "LOAD_FAST"
            and ins[3].opname == "CALL"
            and ins[4].opname == "LOAD_CONST"
            and ins[5].opname == "COMPARE_OP"
            and ins[5].argrepr == ">"
        ):
            min_len = _int_const(ins[4], env)
            if min_len is not None:
                return NativeStage(StageOp.FILTER, NativeOp.FILTER_LENGTH_GT, (min_len,))
        return None

    env = _int_env(func)
    if len(ins) != 5 or ins[0].opname != "LOAD_FAST" or ins[4].opname != "COMPARE_OP":
        return None
    if ins[4].argrepr != "==":
        return None
    mod = _match_mod_bytecode(ins[:3], env)
    eq = _int_const(ins[3], env)
    if mod is not None and eq is not None:
        return NativeStage(StageOp.FILTER, NativeOp.FILTER_MOD_EQ, (mod, eq))
    return None


def compile_flat_map_bytecode(
    func: Callable[..., Any],
    native_kind: NativeKind = "int64",
) -> Optional[NativeStage]:
    code: CodeType = func.__code__
    if code.co_argcount != 1:
        return None
    ins = _bytecode_instructions(func)
    if native_kind == "float64":
        matched = _match_flat_map_range_f64_bytecode(ins, _float_env(func))
        if matched:
            return NativeStage(StageOp.FLAT_MAP, NativeOp.FLAT_MAP_RANGE_F64, matched)
        return None
    if native_kind == "str":
        if (
            len(ins) == 3
            and ins[0].opname == "LOAD_FAST"
            and ins[1].opname == "LOAD_ATTR"
            and ins[1].argval == "split"
            and ins[2].opname == "CALL"
        ):
            return NativeStage(StageOp.FLAT_MAP, NativeOp.FLAT_MAP_SPLIT, ("",))
        return None
    matched = _match_flat_map_range_bytecode(ins, _int_env(func))
    if matched:
        return NativeStage(StageOp.FLAT_MAP, NativeOp.FLAT_MAP_RANGE, matched)
    return None


def compile_reduce_by_key_bytecode(func: Callable[..., Any]) -> Optional[NativeStage]:
    code: CodeType = func.__code__
    if code.co_argcount != 2:
        return None
    ins = _bytecode_instructions(func)
    if len(ins) == 2 and ins[0].opname == "LOAD_FAST_LOAD_FAST" and _is_binop(ins[1], "+"):
        return NativeStage(StageOp.REDUCE_BY_KEY, NativeOp.REDUCE_BY_KEY_SUM, ())
    if (
        len(ins) == 3
        and ins[0].opname == "LOAD_FAST"
        and ins[1].opname == "LOAD_FAST"
        and _is_binop(ins[2], "+")
    ):
        return NativeStage(StageOp.REDUCE_BY_KEY, NativeOp.REDUCE_BY_KEY_SUM, ())
    return None


def try_compile(
    stage_op: StageOp,
    func: Callable[..., Any],
    native_kind: NativeKind = "int64",
) -> Optional[NativeStage]:
    try:
        if stage_op == StageOp.MAP:
            result = compile_map(func, native_kind)
            return result or compile_map_bytecode(func, native_kind)
        if stage_op == StageOp.FILTER:
            result = compile_filter(func, native_kind)
            return result or compile_filter_bytecode(func, native_kind)
        if stage_op == StageOp.FLAT_MAP:
            result = compile_flat_map(func, native_kind)
            return result or compile_flat_map_bytecode(func, native_kind)
        if stage_op == StageOp.REDUCE_BY_KEY:
            result = compile_reduce_by_key(func)
            return result or compile_reduce_by_key_bytecode(func)
    except (OSError, SyntaxError, ValueError):
        if stage_op == StageOp.MAP:
            return compile_map_bytecode(func, native_kind)
        if stage_op == StageOp.FILTER:
            return compile_filter_bytecode(func, native_kind)
        if stage_op == StageOp.FLAT_MAP:
            return compile_flat_map_bytecode(func, native_kind)
        if stage_op == StageOp.REDUCE_BY_KEY:
            return compile_reduce_by_key_bytecode(func)
    return None
