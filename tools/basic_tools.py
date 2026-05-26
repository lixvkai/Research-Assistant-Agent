"""基础工具集 — 用于验证 ReAct 循环。"""

import math
import datetime


def calculator(expression: str) -> str:
    """安全地执行数学计算。"""
    allowed_names = {
        k: v for k, v in math.__dict__.items() if not k.startswith("_")
    }
    allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
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
