"""SSRF / URL 校验单测。"""

import socket

import pytest

from utils.url_safety import validate_public_url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://127.0.0.1:8080/admin",
        "http://[::1]/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/",
        "http://192.168.1.1/",
        "http://172.16.0.5/",
        "file:///etc/passwd",
        "ftp://example.com/x",
        "javascript:alert(1)",
        "",
        "not-a-url",
    ],
)
def test_block_unsafe_urls(url):
    with pytest.raises(ValueError):
        validate_public_url(url)


def test_block_localhost_hostname(monkeypatch):
    def fake_getaddrinfo(host, *args, **kwargs):
        assert host == "localhost"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(ValueError, match="内网"):
        validate_public_url("http://localhost/secret")


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/",
        "http://example.com/path?q=1",
        "https://arxiv.org/abs/1706.03762",
    ],
)
def test_allow_public_urls(url, monkeypatch):
    def fake_getaddrinfo(host, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    assert validate_public_url(url) == url


def test_block_hostname_resolving_to_private(monkeypatch):
    def fake_getaddrinfo(host, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(ValueError, match="内网"):
        validate_public_url("http://evil.example/x")


def test_fetch_webpage_rejects_ssrf():
    from tools.web_tool import fetch_webpage

    out = fetch_webpage("http://127.0.0.1:1/")
    assert "抓取网页出错" in out
    assert "不安全" in out
