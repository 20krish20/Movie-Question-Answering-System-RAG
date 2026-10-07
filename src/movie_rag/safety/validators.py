"""
Validation for LLM-generated pandas code before it is executed.

The app is public, so the generated line is treated as untrusted input.
Instead of blocklisting dangerous strings, the code is parsed and every AST
node must be on an allowlist:

- exactly one statement: `result = <expression>`
- names: only `rich_movies` and a few pure builtins (len, round, ...)
- attributes: only known read-only pandas methods/accessors, never `_private`
- no lambdas, comprehensions, f-strings, `**`, imports or keyword unpacking
- no `inplace=` (the DataFrame is shared across requests)
- multiplication is bounded so strings/Series can't be blown up in memory
"""
import ast
import builtins

MAX_CODE_LEN = 600
MAX_STR_CONST_LEN = 200
MAX_INT_MULTIPLIER = 100

SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in ["len", "round", "int", "float", "str", "bool", "abs", "min", "max", "sum", "sorted", "list"]
}
ALLOWED_NAMES = {"rich_movies", *SAFE_BUILTINS}

ALLOWED_ATTRS = {
    # indexing / shape
    "loc", "iloc", "at", "iat", "index", "columns", "shape", "size", "empty", "values", "name", "dtype", "T",
    # selection / ordering
    "head", "tail", "nlargest", "nsmallest", "sort_values", "sort_index", "drop_duplicates", "dropna", "drop",
    "reset_index", "set_index", "rename", "isin", "between", "isna", "notna", "isnull", "notnull", "where", "mask",
    "filter", "fillna", "replace", "astype", "copy", "assign", "to_frame", "explode", "clip", "get",
    # aggregation
    "groupby", "agg", "aggregate", "mean", "median", "sum", "count", "min", "max", "std", "var", "prod",
    "nunique", "unique", "value_counts", "idxmax", "idxmin", "describe", "quantile", "corr", "mode", "rank",
    "cumsum", "first", "last", "any", "all", "apply", "map", "transform", "round", "abs",
    # conversion
    "tolist", "to_list", "to_dict", "to_numpy", "item", "iloc",
    # string accessor (no pad/repeat/center/ljust/rjust/zfill: those can allocate unbounded memory)
    "str", "contains", "startswith", "endswith", "lower", "upper", "title", "strip", "lstrip", "rstrip",
    "split", "len", "match", "fullmatch", "extract", "findall", "slice",
}

_ALLOWED_NODES = (
    ast.Module, ast.Assign, ast.Expression,
    ast.Name, ast.Load, ast.Store, ast.Attribute, ast.Subscript, ast.Slice, ast.Call, ast.keyword, ast.Constant,
    ast.Compare, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.IfExp, ast.List, ast.Tuple, ast.Dict,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
    ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd, ast.Invert,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.BitAnd, ast.BitOr,
)


class UnsafeCodeError(ValueError):
    pass


def _check_mult(node: ast.BinOp):
    # Allow `x * <number>` only: no chained/nested multiplication, ints capped, so
    # `rich_movies["Title"] * 10**9` style memory bombs are rejected.
    operands = [node.left, node.right]
    for op in operands:
        if any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult) for n in ast.walk(op)):
            raise UnsafeCodeError("Nested multiplication is not allowed")
    consts = [op for op in operands if isinstance(op, ast.Constant) and isinstance(op.value, (int, float))
              and not isinstance(op.value, bool)]
    if not consts:
        raise UnsafeCodeError("Multiplication must have a numeric constant operand")
    for c in consts:
        if isinstance(c.value, int) and abs(c.value) > MAX_INT_MULTIPLIER:
            raise UnsafeCodeError(f"Integer multipliers above {MAX_INT_MULTIPLIER} are not allowed")


def validate_code(code: str) -> ast.Module:
    code = code.strip()

    if len(code) > MAX_CODE_LEN:
        raise UnsafeCodeError("Generated code is too long")
    if "\n" in code:
        raise UnsafeCodeError("Factual code must be a SINGLE LINE: result = <expression>")
    if "..." in code:
        raise UnsafeCodeError("Ellipsis (...) not allowed")

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise UnsafeCodeError(f"Generated code is not valid Python: {e.msg}") from e

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        raise UnsafeCodeError("Code must match: result = <expression>")
    assign = tree.body[0]
    if len(assign.targets) != 1 or not (isinstance(assign.targets[0], ast.Name) and assign.targets[0].id == "result"):
        raise UnsafeCodeError("Code must match: result = <expression>")

    uses_df = False
    for node in ast.walk(assign.value):
        if not isinstance(node, _ALLOWED_NODES):
            raise UnsafeCodeError(f"Forbidden construct in generated code: {type(node).__name__}")

        if isinstance(node, ast.Name):
            if node.id not in ALLOWED_NAMES or not isinstance(node.ctx, ast.Load):
                raise UnsafeCodeError(f"Forbidden name: {node.id}")
            uses_df |= node.id == "rich_movies"

        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr not in ALLOWED_ATTRS:
                raise UnsafeCodeError(f"Forbidden attribute: .{node.attr}")

        elif isinstance(node, ast.keyword):
            if node.arg is None:
                raise UnsafeCodeError("Keyword unpacking (**) is not allowed")
            if node.arg == "inplace":
                raise UnsafeCodeError("inplace= is not allowed")

        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str) and len(node.value) > MAX_STR_CONST_LEN:
                raise UnsafeCodeError("String constant too long")
            if isinstance(node.value, bytes):
                raise UnsafeCodeError("Bytes constants are not allowed")

        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            _check_mult(node)

    if not uses_df:
        raise UnsafeCodeError("Code must reference rich_movies")

    return tree
