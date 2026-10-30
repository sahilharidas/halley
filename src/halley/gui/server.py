"""The GUI's web server: the pages under ``static/`` and the JSON API those pages call.

:func:`make_server` binds a :class:`GuiServer` to an address, the loopback interface unless told
otherwise, with the API's routes and a token. Every API request must carry the token in the
``X-Halley-Token`` header and name the server's own address in its ``Host`` header, or it is
refused before any route runs. The browser is given the token in the fragment of the address it
opens (``/#token=...``), which a browser sends to no server. A page of another site open in the
same browser therefore cannot use the API: it cannot read the token, it cannot send the header
to another origin without a CORS preflight, which this server does not answer, and through a
host name rebound to the loopback address its requests name that host, not this one. The pages
themselves hold no data, so they are served without the token, under a content security policy
that loads nothing from anywhere else.

A route is a function of a :class:`Request` that returns what the response's JSON holds, and
refuses a request by raising :class:`ApiError`. :func:`jsonable` turns what a route returns into
what ``json.dumps`` writes exactly: NumPy values as Python ones, and a float that is not finite as
a string, since JSON has no NaN.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import http.server
import ipaddress
import json
import math
import secrets
import socket
import sys
import traceback
import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any, cast

import numpy as np

STATIC_DIR = Path(__file__).resolve().parent / "static"

STATIC_FILES: Mapping[str, tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/chart.js": ("chart.js", "text/javascript; charset=utf-8"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
}
"""The pages and their media types by the path each is served at. Nothing else under ``static/``
is served, and no part of a request's path is joined to a directory."""

TOKEN_HEADER = "X-Halley-Token"

MAX_BODY = 1 << 20
"""The largest request body the API reads, in bytes: a request names files and settings, not
data."""

CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)

LOOPBACK = ("127.0.0.1", "localhost", "[::1]")
"""The loopback names a ``Host`` header may give, with the server's port."""

NON_FINITE = {math.inf: "Infinity", -math.inf: "-Infinity"}


@dataclass(frozen=True, slots=True)
class Request:
    """An API request as a route sees it: its method, its path, its query parameters (the last
    value of each) and its JSON body, an empty mapping for a GET."""

    method: str
    path: str
    query: Mapping[str, str]
    body: Mapping[str, object]

    def arg(self, name: str) -> str:
        """A query parameter the request must give."""
        value = self.query.get(name, "")
        if value == "":
            raise ApiError(HTTPStatus.BAD_REQUEST, f"the request gives no {name}")
        return value

    def int_arg(self, name: str, default: int | None = None) -> int:
        """A whole-number query parameter, or ``default`` when the request gives none; with no
        default the request must give it."""
        value = self.query.get(name, "")
        if value == "":
            if default is None:
                raise ApiError(HTTPStatus.BAD_REQUEST, f"the request gives no {name}")
            return default
        try:
            return int(value)
        except ValueError:
            raise ApiError(
                HTTPStatus.BAD_REQUEST, f"{name} is a whole number, not {value!r}"
            ) from None


class ApiError(Exception):
    """A request refused: the HTTP status, a message for the page and details the page may
    show, such as the model source location a build error points at."""

    def __init__(
        self, status: int, message: str, detail: Mapping[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.status = int(status)
        self.message = message
        self.detail: dict[str, object] = dict(detail or {})


Route = Callable[[Request], object]
"""What answers one method and path of the API."""


def jsonable(value: object) -> object:
    """``value`` as what ``json.dumps`` writes without loss or failure: NumPy scalars and arrays as
    Python numbers and lists, a float that is not finite as the string JavaScript prints for it
    (``"NaN"``, ``"Infinity"``, ``"-Infinity"``), dates as ISO strings, paths as strings, tuples
    and sets as lists and mapping keys as strings. A finite float keeps every bit: ``json.dumps``
    writes the shortest text that reads back as the same float, and so does JavaScript."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, float):  # a NumPy float64 too, which is a float
        number = float(value)
        if math.isfinite(number):
            return number
        return NON_FINITE.get(number, "NaN")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, np.ndarray):
        array = cast("np.ndarray[Any, Any]", value)
        if array.dtype.kind == "f" and bool(np.isfinite(array).all()):
            return array.tolist()  # the common case, without a Python call per value
        if array.dtype.kind in "iub":
            return array.tolist()
        return jsonable(array.tolist())
    if isinstance(value, np.generic):
        return jsonable(cast("np.generic[Any]", value).item())
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        items = cast(Mapping[object, object], value).items()
        return {str(k): jsonable(v) for k, v in items}
    if isinstance(value, (list, tuple, set, frozenset)):
        seq = cast(list[object] | tuple[object, ...] | set[object] | frozenset[object], value)
        return [jsonable(v) for v in seq]
    raise TypeError(f"no JSON form for a {type(value).__name__}")


class GuiServer(http.server.ThreadingHTTPServer):
    """The GUI's server: it serves the pages and answers the API on a thread per request."""

    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second socket bind an address already in use, so a second
    # server would share the port without a word; elsewhere it only skips TIME_WAIT.
    allow_reuse_address = sys.platform != "win32"

    def __init__(
        self,
        address: tuple[str, int],
        routes: Mapping[tuple[str, str], Route],
        token: str,
        extra_hosts: tuple[str, ...] = (),
    ) -> None:
        self.routes: dict[tuple[str, str], Route] = dict(routes)
        self.token = token
        self.extra_hosts = tuple(extra_hosts)
        if ":" in address[0]:
            self.address_family = socket.AF_INET6  # an IPv6 address, such as ::1
        super().__init__(address, _Handler)

    @property
    def host(self) -> str:
        return str(self.server_address[0])

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    def browser_url(self) -> str:
        """The address to open: the server's own, on the loopback interface when it listens on
        every interface, with the token in the fragment."""
        host = self.host
        if host in ("", "0.0.0.0"):
            host = "127.0.0.1"
        elif host == "::":
            host = "[::1]"  # an IPv6 socket on Windows takes no IPv4 connection by default
        elif ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{self.port}/#token={self.token}"

    def hosts(self) -> frozenset[str]:
        """The ``Host`` headers the server answers: the loopback names, the address it is bound
        to and any extra names it was given, each with its port, in lower case; on port 80 also
        without it, since a client leaves the default port out."""
        names = {_bracketed(n) for n in (*LOOPBACK, self.host, *self.extra_hosts)}
        out = {f"{name}:{self.port}".lower() for name in names}
        if self.port == 80:
            out |= {name.lower() for name in names}
        return frozenset(out)

    def answers(self, header: str) -> bool:
        """Whether a request whose ``Host`` header is ``header`` is for this server: one of
        :meth:`hosts`, or, for a server listening on every interface, any IP address with the
        server's port. A page of another site that reaches the server by rebinding a name it
        controls still sends that name, never an address, so an address carries no such page,
        and the machine's addresses on every network it is on are its own."""
        text = header.strip().lower()
        if text in self.hosts():
            return True
        if self.host not in ("0.0.0.0", "::", ""):
            return False
        if text.startswith("["):
            address, _, rest = text[1:].partition("]")
            port = rest[1:] if rest.startswith(":") else None if not rest else "?"
        else:
            address, colon, port_text = text.rpartition(":")
            if not colon:
                address, port_text = text, ""
            port = port_text or None
        if port != (str(self.port) if port is not None else None):
            return False
        if port is None and self.port != 80:
            return False
        try:
            ipaddress.ip_address(address)
        except ValueError:
            return False
        return True


def _bracketed(name: str) -> str:
    """An IPv6 address as a ``Host`` header writes it, in brackets."""
    return f"[{name}]" if ":" in name and not name.startswith("[") else name


_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError)
"""What writing to, or reading from, a client that has gone raises."""


class _Handler(http.server.BaseHTTPRequestHandler):
    server_version = "Halley-GUI"
    sys_version = ""
    timeout = 30
    """Seconds a connection may stay silent before it is closed: one that sends nothing, or never
    finishes its headers, would otherwise hold a thread for ever."""

    raw_body: bytes = b""
    """The request's body, read before any check answers: a refusal sent while the body is still
    unread is lost on Windows, which resets the connection when it closes with data unread."""

    @property
    def gui(self) -> GuiServer:
        return cast(GuiServer, self.server)

    def log_message(self, format: str, *args: Any) -> None:
        """Requests are not logged: the terminal shows the address to open, and errors reach the
        page."""

    # http.server calls do_<METHOD>; a method without one is answered 501, CORS preflights
    # (OPTIONS) included, so no other origin can call the API from a page.
    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        url = urllib.parse.urlsplit(self.path)
        try:
            self.raw_body = self._read_body() if method == "POST" else b""
            if not self.gui.answers(self.headers.get("Host") or ""):
                raise ApiError(
                    HTTPStatus.FORBIDDEN, "this server answers requests to its own address only"
                )
            if url.path.startswith("/api/"):
                self._api(method, url.path, url.query)
            elif method == "GET" and url.path in STATIC_FILES:
                self._static(*STATIC_FILES[url.path])
            else:
                raise ApiError(HTTPStatus.NOT_FOUND, f"there is no page {url.path}")
        except ApiError as exc:
            body = {"error": {"message": exc.message, **exc.detail}}
            with contextlib.suppress(*_GONE):  # a client that left needs no answer
                self._send(exc.status, _encode(body), "application/json; charset=utf-8")
        except _GONE:
            pass

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            raise ApiError(HTTPStatus.BAD_REQUEST, "the Content-Length is not a number") from None
        if length < 0 or length > MAX_BODY:
            self.close_connection = True
            raise ApiError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"the body is {length} bytes; the API reads at most {MAX_BODY}",
            )
        return self.rfile.read(length) if length else b""

    def _api(self, method: str, path: str, query: str) -> None:
        token = self.headers.get(TOKEN_HEADER) or ""
        if not secrets.compare_digest(token.encode(), self.gui.token.encode()):
            raise ApiError(
                HTTPStatus.FORBIDDEN,
                "the request carries no valid session token; open the address the server "
                "printed when it started",
            )
        route = self.gui.routes.get((method, path))
        if route is None:
            if any(known == path for _method, known in self.gui.routes):
                raise ApiError(HTTPStatus.METHOD_NOT_ALLOWED, f"{path} does not take a {method}")
            raise ApiError(HTTPStatus.NOT_FOUND, f"the API has no {path}")
        body = self._body() if method == "POST" else {}
        params = dict(urllib.parse.parse_qsl(query, keep_blank_values=True))
        try:
            raw = _encode(route(Request(method, path, params, body)))
        except ApiError:
            raise
        except Exception as exc:  # a fault in the GUI or the engine: the page shows all of it
            raise ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"{type(exc).__name__}: {exc}",
                {"type": type(exc).__name__, "traceback": traceback.format_exc()},
            ) from exc
        self._send(HTTPStatus.OK, raw, "application/json; charset=utf-8")

    def _body(self) -> dict[str, object]:
        if self.headers.get_content_type() != "application/json":
            raise ApiError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "the API reads a JSON body")
        raw = self.raw_body or b"{}"
        try:
            body: object = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"the body is not JSON: {exc}") from None
        if not isinstance(body, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "the body is a JSON object")
        return cast(dict[str, object], body)

    def _static(self, name: str, media_type: str) -> None:
        raw = (STATIC_DIR / name).read_bytes()
        self._send(HTTPStatus.OK, raw, media_type, static=True)

    def _send(self, status: int, raw: bytes, media_type: str, static: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-cache" if static else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if static:
            self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
            self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(raw)


def _encode(value: object) -> bytes:
    return json.dumps(
        jsonable(value), allow_nan=False, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def make_server(
    routes: Mapping[tuple[str, str], Route],
    host: str = "127.0.0.1",
    port: int = 0,
    token: str | None = None,
    extra_hosts: tuple[str, ...] = (),
) -> GuiServer:
    """A server bound to ``host:port`` that answers ``routes``, by ``(method, path)``, to requests
    that carry ``token``, a new random one by default. Port 0 takes a free port the system picks.
    ``serve_forever()`` runs it; ``extra_hosts`` are further names its ``Host`` header check
    accepts, for a server reached by another name than its address."""
    return GuiServer((host, port), routes, token or secrets.token_urlsafe(32), extra_hosts)
