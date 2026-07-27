"""路径安全校验 — 防止目录穿越与任意文件读写。"""

from __future__ import annotations

import os


def resolve_under(base_dir: str, path: str) -> str:
    """将 path 解析为绝对路径，并确保落在 base_dir 内。

    Raises:
        ValueError: 路径为空，或逃出 base_dir。
    """
    if not path or not str(path).strip():
        raise ValueError("路径不能为空")

    base = os.path.realpath(base_dir)
    candidate = path if os.path.isabs(path) else os.path.join(base, path)
    resolved = os.path.realpath(candidate)

    if resolved != base and not resolved.startswith(base + os.sep):
        raise ValueError(f"路径不允许逃出工作目录：{path}")
    return resolved


def safe_filename(filename: str) -> str:
    """只保留文件名本身，拒绝含路径分隔符或特殊名的输入。"""
    if not filename or not str(filename).strip():
        raise ValueError("文件名不能为空")
    raw = filename.strip()
    # 拒绝任何路径成分
    if "/" in raw or "\\" in raw or raw in (".", ".."):
        raise ValueError(f"非法文件名：{filename}")
    name = os.path.basename(raw)
    if name != raw or name in (".", ".."):
        raise ValueError(f"非法文件名：{filename}")
    return name
