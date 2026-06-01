"""calculator 安全求值测试：正确性 + 防 DoS + 防注入。"""

import pytest

from tools.basic_tools import calculator


@pytest.mark.parametrize("expr,expected", [
    ("2**10", "1024"),
    ("sqrt(144)", "12"),
    ("max(1, 2, 3)", "= 3"),
    ("abs(-5)", "= 5"),
    ("1 + 2 * 3", "= 7"),
    ("10 // 3", "= 3"),
])
def test_calculator_correct(expr, expected):
    assert expected in calculator(expr)


def test_calculator_blocks_huge_power():
    out = calculator("2**99999999")
    assert "出错" in out and "幂指数" in out


@pytest.mark.parametrize("evil", [
    "__import__('os').system('echo hi')",
    "(1).__class__",
    "open('x')",
    "[].append(1)",
    "lambda: 1",
    "eval('1')",
])
def test_calculator_blocks_injection(evil):
    assert "出错" in calculator(evil)
