"""路径穿越 / 安全文件名单测。"""

import os
from pathlib import Path

import pytest

from utils.path_safety import resolve_under, safe_filename


def test_safe_filename_ok():
    assert safe_filename("paper.pdf") == "paper.pdf"
    assert safe_filename("  notes.txt  ") == "notes.txt"


@pytest.mark.parametrize(
    "name",
    [
        "../etc/passwd",
        "..\\windows\\system32",
        "/etc/passwd",
        "foo/bar.pdf",
        "foo\\bar.pdf",
        ".",
        "..",
        "",
        "   ",
    ],
)
def test_safe_filename_rejects(name):
    with pytest.raises(ValueError):
        safe_filename(name)


def test_resolve_under_allows_inside(tmp_path: Path):
    base = tmp_path / "papers"
    base.mkdir()
    target = base / "a.pdf"
    target.write_text("x", encoding="utf-8")

    assert resolve_under(str(base), str(target)) == os.path.realpath(target)
    assert resolve_under(str(base), "a.pdf") == os.path.realpath(target)


def test_resolve_under_blocks_escape(tmp_path: Path):
    base = tmp_path / "papers"
    base.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError):
        resolve_under(str(base), str(outside))
    with pytest.raises(ValueError):
        resolve_under(str(base), "../secret.txt")


def test_ingest_paper_rejects_outside_path(tmp_path: Path, monkeypatch):
    from tools import rag_tool

    papers = tmp_path / "papers"
    papers.mkdir()
    monkeypatch.setattr(rag_tool, "PAPERS_DIR", str(papers))

    outside = tmp_path / "evil.txt"
    outside.write_text("nope", encoding="utf-8")

    out = rag_tool.ingest_paper(str(outside))
    assert out.startswith("导入失败")
    # 不应触发引擎创建去碰 chromadb
    assert rag_tool._engine is None


def test_delete_path_guard_same_as_app(tmp_path: Path):
    """回归：删除流程使用的 safe_filename + resolve_under 能挡住穿越。"""
    papers = tmp_path / "papers"
    papers.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError):
        name = safe_filename("../victim.txt")
        resolve_under(str(papers), name)

    assert victim.exists()
