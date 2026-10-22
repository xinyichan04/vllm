# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-connection receive-path tracing, enabled by ``VLLM_CONN_TRACE=<path>``.

Writes one JSON line per event so the life of a request can be reconstructed
from the earliest point vLLM can observe it up to the last request byte:

    listen_armed      listening socket created with TCP_SAVE_SYN
    syn_seen          a SYN for the listening port, seen live on a raw
                      AF_PACKET socket (needs CAP_NET_RAW in the socket's
                      network namespace); carries the kernel's receive
                      timestamp and the time this process saw it
    factory           uvicorn protocol constructed; uvloop calls this from its
                      listen callback *before* accept4(), so it is the first
                      Python code that runs for a new connection
    connection_made   socket accepted; the kernel's saved SYN headers and
                      TCP_INFO are read here, which back-dates the handshake
    first_data        first data_received() call (request line bytes)
    message_begin     HTTP parser saw the start of a request
    headers_complete  request line + headers parsed; ASGI task is scheduled
    asgi_entered      the ASGI app (vLLM's FastAPI stack) starts running
    message_complete  last request byte parsed: the full request is in
    connection_lost   connection closed

The kernel handles the handshake itself, so the TCP stack never tells the
server about a SYN. Two ways around that are used here: a packet socket sees
the SYN on the interface as it arrives (syn_seen), and TCP_SAVE_SYN makes the
kernel keep the SYN's IP+TCP headers so they can be read back at accept time,
with ``tcpi_last_ack_recv`` (jiffies resolution) dating the final handshake
ACK. The second needs no privileges and is the fallback when the first is not
permitted.
"""

import itertools
import json
import os
import socket
import struct
import threading
import time
from typing import Any

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# From linux/tcp.h; not exported by the socket module.
TCP_SAVE_SYN = 27
TCP_SAVED_SYN = 28

# Key under which the connection id is placed in the ASGI scope, so
# asgi_entered can be joined to the connection-level events.
SCOPE_KEY = "vllm.conn_trace_id"

_conn_ids = itertools.count(1)
_fh = None
_lock = threading.Lock()


def _emit(ev: str, conn: int | None, **fields: Any) -> None:
    global _fh
    if _fh is None:
        path = envs.VLLM_CONN_TRACE
        assert path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        _fh = open(path, "a", buffering=1 << 16)  # noqa: SIM115
    rec = {
        "ev": ev,
        "conn": conn,
        "pid": os.getpid(),
        "t": time.time(),
        "ns": time.perf_counter_ns(),
        **fields,
    }
    with _lock:
        _fh.write(json.dumps(rec) + "\n")


def _flush() -> None:
    if _fh is not None:
        _fh.flush()


ETH_P_ALL = 0x0003
SO_ATTACH_FILTER = 26
SO_TIMESTAMPNS = 35


def _syn_filter(port: int) -> bytes:
    """Classic BPF: IPv4 TCP, not a fragment, dst port == port, SYN without ACK."""
    insns = [
        (0x28, 0, 0, 12),  # ldh [12]            ethertype
        (0x15, 0, 11, 0x0800),  # jne IPv4 -> drop
        (0x30, 0, 0, 23),  # ldb [23]            IP protocol
        (0x15, 0, 9, 6),  # jne TCP -> drop
        (0x28, 0, 0, 20),  # ldh [20]            fragment offset
        (0x45, 7, 0, 0x1FFF),  # jset -> drop
        (0xB1, 0, 0, 14),  # ldxb 4*([14]&0xf)  IP header length
        (0x48, 0, 0, 16),  # ldh [x+16]         TCP dst port
        (0x15, 0, 4, port),  # jne port -> drop
        (0x50, 0, 0, 27),  # ldb [x+27]         TCP flags
        (0x54, 0, 0, 0x12),  # and SYN|ACK
        (0x15, 0, 1, 0x02),  # jne SYN-only -> drop
        (0x06, 0, 0, 0x40000),  # accept
        (0x06, 0, 0, 0),  # drop
    ]
    return b"".join(struct.pack("HBBI", *i) for i in insns)


def _watch_syns(raw: socket.socket) -> None:
    while True:
        try:
            data, anc, _flags, addr = raw.recvmsg(128, 64)
        except OSError:
            return
        t_seen = time.time()
        # Loopback shows each packet twice; keep the arriving copy.
        if addr[2] == socket.PACKET_OUTGOING:
            continue
        t_kernel = None
        for _level, kind, cdata in anc:
            if kind == SO_TIMESTAMPNS:
                sec, nsec = struct.unpack("qq", cdata[:16])
                t_kernel = sec + nsec / 1e9
        ihl = (data[14] & 0x0F) * 4
        syn = _parse_syn(data[14:])
        tcp = syn.get("tcp") or {}
        _emit(
            "syn_seen",
            None,
            iface=addr[0],
            peer=[socket.inet_ntoa(data[26:30]), tcp.get("sport")],
            seq=tcp.get("seq"),
            t_kernel=t_kernel,
            ip_hdr_len=ihl,
        )


def start_syn_watcher(port: int) -> bool:
    """Watch for SYNs to `port` on a raw packet socket, in a daemon thread.

    Needs CAP_NET_RAW in this network namespace; returns False without it.
    The BPF filter runs in the kernel, so only matching SYNs wake the thread.
    """
    try:
        raw = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P_ALL))
    except (OSError, AttributeError) as exc:
        logger.warning("conn_trace: live SYN capture unavailable: %s", exc)
        return False
    import ctypes

    prog = _syn_filter(port)
    buf = ctypes.create_string_buffer(prog)
    fprog = struct.pack("HL", len(prog) // 8, ctypes.addressof(buf))
    raw.setsockopt(socket.SOL_SOCKET, SO_ATTACH_FILTER, fprog)
    raw.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
    threading.Thread(
        target=_watch_syns, args=(raw,), name="conn-trace-syn", daemon=True
    ).start()
    return True


def enable_save_syn(sock: socket.socket) -> None:
    """Ask the kernel to keep each incoming SYN's headers for this listener."""
    try:
        sock.setsockopt(socket.IPPROTO_TCP, TCP_SAVE_SYN, 1)
        ok = True
    except OSError as exc:
        logger.warning("conn_trace: TCP_SAVE_SYN unavailable: %s", exc)
        ok = False
    addr = sock.getsockname()
    live = start_syn_watcher(addr[1])
    _emit("listen_armed", None, addr=list(addr), save_syn=ok, live_syn=live)
    _flush()


def _parse_syn(raw: bytes) -> dict[str, Any]:
    """Decode the saved IPv4/IPv6 + TCP headers of the SYN."""
    out: dict[str, Any] = {"raw_hex": raw.hex(), "len": len(raw)}
    if not raw:
        return out
    version = raw[0] >> 4
    if version == 4:
        ihl = (raw[0] & 0x0F) * 4
        out["ip"] = {"v": 4, "ttl": raw[8], "id": struct.unpack_from("!H", raw, 4)[0]}
        tcp = raw[ihl:]
    elif version == 6:
        out["ip"] = {"v": 6, "hop_limit": raw[7]}
        tcp = raw[40:]
    else:
        return out
    if len(tcp) < 20:
        return out
    sport, dport, seq, _ack, off_flags, win = struct.unpack_from("!HHIIHH", tcp)
    doff = (off_flags >> 12) * 4
    flags = off_flags & 0x1FF
    opts: dict[str, Any] = {}
    i, o = 0, tcp[20:doff]
    while i < len(o):
        kind = o[i]
        if kind == 0:
            break
        if kind == 1:
            i += 1
            continue
        if i + 1 >= len(o):
            break
        length = o[i + 1]
        body = o[i + 2 : i + length]
        if kind == 2 and len(body) == 2:
            opts["mss"] = struct.unpack("!H", body)[0]
        elif kind == 3 and len(body) == 1:
            opts["wscale"] = body[0]
        elif kind == 4:
            opts["sack_ok"] = True
        elif kind == 8 and len(body) == 8:
            opts["tsval"], opts["tsecr"] = struct.unpack("!II", body)
        i += max(length, 2)
    out["tcp"] = {
        "sport": sport,
        "dport": dport,
        "seq": seq,
        "flags": f"0x{flags:03x}",
        "is_syn": bool(flags & 0x002) and not flags & 0x010,
        "window": win,
        "options": opts,
    }
    return out


def _tcp_info(sock: Any) -> dict[str, Any] | None:
    """The fields of struct tcp_info (linux/tcp.h) that date the connection."""
    try:
        b = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 256)
    except (OSError, AttributeError):
        return None
    if len(b) < 144:
        return None
    u32 = struct.unpack_from("<24I", b, 8)
    return {
        "state": b[0],
        "last_data_recv_ms": u32[11],
        "last_ack_recv_ms": u32[12],
        "rtt_us": u32[15],
        "rttvar_us": u32[16],
        "bytes_received": struct.unpack_from("<Q", b, 128)[0],
        "segs_in": struct.unpack_from("<I", b, 140)[0],
    }


def traced_protocol_class(base: type) -> type:
    """Subclass uvicorn's HTTP protocol class to emit receive-path events.

    The parser callbacks (on_message_begin etc.) exist on HttpToolsProtocol;
    with h11 only the connection-level events are emitted.
    """

    class TracedProtocol(base):  # type: ignore[valid-type,misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._ct_id = next(_conn_ids)
            self._ct_req = 0
            self._ct_seen_data = False
            _emit("factory", self._ct_id)
            super().__init__(*args, **kwargs)

        def connection_made(self, transport: Any) -> None:
            t_now = time.time()
            sock = transport.get_extra_info("socket")
            fields: dict[str, Any] = {
                "peer": list(transport.get_extra_info("peername") or ()),
            }
            if sock is not None:
                try:
                    raw = sock.getsockopt(socket.IPPROTO_TCP, TCP_SAVED_SYN, 512)
                    fields["syn"] = _parse_syn(raw)
                except OSError as exc:
                    fields["syn"] = {"error": str(exc)}
                info = _tcp_info(sock)
                fields["tcp_info"] = info
                if info is not None:
                    # The child socket's rcv_tstamp is set when the final
                    # handshake ACK arrives; SYN ~= that minus one RTT.
                    t_hs = t_now - info["last_ack_recv_ms"] / 1000
                    fields["est_handshake_done_t"] = t_hs
                    fields["est_syn_t"] = t_hs - info["rtt_us"] / 1e6
            _emit("connection_made", self._ct_id, **fields)
            super().connection_made(transport)

        def data_received(self, data: bytes) -> None:
            if not self._ct_seen_data:
                self._ct_seen_data = True
                _emit(
                    "first_data",
                    self._ct_id,
                    nbytes=len(data),
                    head=data[:80].split(b"\r\n", 1)[0].decode("latin-1"),
                )
            super().data_received(data)

        def on_message_begin(self) -> None:
            super().on_message_begin()
            self._ct_req += 1
            self._ct_body = 0
            self._ct_chunks = 0
            self.scope[SCOPE_KEY] = (self._ct_id, self._ct_req)  # type: ignore[literal-required]
            _emit("message_begin", self._ct_id, req=self._ct_req)

        def on_headers_complete(self) -> None:
            super().on_headers_complete()
            cl = dict(self.scope.get("headers") or ()).get(b"content-length")
            _emit(
                "headers_complete",
                self._ct_id,
                req=self._ct_req,
                method=self.scope.get("method"),
                path=self.scope.get("path"),
                content_length=int(cl) if cl else None,
            )

        def on_body(self, body: bytes) -> None:
            self._ct_body += len(body)
            self._ct_chunks += 1
            super().on_body(body)

        def on_message_complete(self) -> None:
            _emit(
                "message_complete",
                self._ct_id,
                req=self._ct_req,
                body_bytes=self._ct_body,
                body_chunks=self._ct_chunks,
            )
            _flush()
            super().on_message_complete()

        def connection_lost(self, exc: Exception | None) -> None:
            _emit("connection_lost", self._ct_id, error=repr(exc) if exc else None)
            _flush()
            super().connection_lost(exc)

    TracedProtocol.__name__ = f"Traced{base.__name__}"
    return TracedProtocol


def wrap_app(app: Any) -> Any:
    """Outermost ASGI wrapper marking when the request enters vLLM's app."""

    async def traced_app(scope, receive, send):
        if scope["type"] == "http":
            conn, req = scope.get(SCOPE_KEY, (None, None))
            _emit("asgi_entered", conn, req=req, path=scope.get("path"))
        await app(scope, receive, send)

    return traced_app
