"""URL 安全校验 — 防止 SSRF（内网 / 元数据 / 危险 scheme）。"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


_BLOCKED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
        return True
    return any(ip in net for net in _BLOCKED_NETWORKS)


def validate_public_url(url: str) -> str:
    """校验 URL 可安全抓取；通过则返回规范化 URL，否则抛 ValueError。"""
    if not url or not isinstance(url, str):
        raise ValueError("URL 不能为空")

    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"仅允许 http/https：{parsed.scheme or '(空)'}")
    if not parsed.hostname:
        raise ValueError("URL 缺少主机名")

    host = parsed.hostname
    # 字面量 IP
    try:
        ip = ipaddress.ip_address(host)
        if _is_blocked_ip(ip):
            raise ValueError(f"禁止访问内网/保留地址：{host}")
        return url
    except ValueError as e:
        if "禁止访问" in str(e):
            raise
        # 不是字面量 IP，继续 DNS 解析

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise ValueError(f"无法解析主机名：{host}") from e

    if not infos:
        raise ValueError(f"无法解析主机名：{host}")

    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if _is_blocked_ip(ip):
            raise ValueError(f"主机解析到内网/保留地址，已拒绝：{host} → {addr}")

    return url
