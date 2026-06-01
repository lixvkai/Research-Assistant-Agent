"""基础工具集 — 用于验证 ReAct 循环。"""

import ast
import math
import datetime
import operator

# 基于 AST 的安全数值求值：白名单运算符 / 函数 / 常量，杜绝 eval 的注入与 DoS 风险。

_MAX_POW_EXPONENT = 1000  # 限制幂指数，防止 2**99999999 这类内存/CPU 炸弹

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: None,  # 特殊处理，见 _eval_node
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_ALLOWED_FUNCS = {
    k: v for k, v in math.__dict__.items()
    if not k.startswith("_") and callable(v)
}
_ALLOWED_FUNCS.update({"abs": abs, "round": round, "min": min, "max": max})
_ALLOWED_NAMES = {
    k: v for k, v in math.__dict__.items()
    if not k.startswith("_") and isinstance(v, (int, float))
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("仅支持数值常量")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"不支持的运算符：{op_type.__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if op_type is ast.Pow:
            if abs(right) > _MAX_POW_EXPONENT:
                raise ValueError(f"幂指数过大（上限 {_MAX_POW_EXPONENT}）")
            return left ** right
        return _BIN_OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"不支持的一元运算符：{op_type.__name__}")
        return _UNARY_OPS[op_type](_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ValueError("不支持的函数调用")
        if node.keywords:
            raise ValueError("不支持关键字参数")
        args = [_eval_node(a) for a in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"未知标识符：{node.id}")
    raise ValueError(f"不支持的表达式节点：{type(node).__name__}")


def calculator(expression: str) -> str:
    """安全地执行数学计算（AST 白名单求值，不使用 eval）。"""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算出错：{e}"


def get_current_time() -> str:
    """获取当前日期和时间。"""
    now = datetime.datetime.now()
    return f"当前时间：{now.strftime('%Y年%m月%d日 %H:%M:%S')}，星期{['一','二','三','四','五','六','日'][now.weekday()]}"


TOOL_DEFINITIONS = [
    {
        "name": "calculator",
        "description": "执行数学计算。支持基本运算和数学函数（sin, cos, sqrt, log, pi 等）。",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，例如 '2**10' 或 'sqrt(144)' 或 'sin(pi/4)'",
                }
            },
            "required": ["expression"],
        },
        "func": calculator,
    },
    {
        "name": "get_current_time",
        "description": "获取当前日期和时间信息。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "func": get_current_time,
    },
]
