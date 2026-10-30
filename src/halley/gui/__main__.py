"""``python -m halley.gui``: start the GUI's server and open the GUI in the browser.

The server listens on the loopback interface at port 8765, or at a free port when that one is
taken, and prints the address to open, which carries the session's token. Ctrl+C stops it; a
build or run that has started finishes first, since the engine cannot stop one part way.
"""

from __future__ import annotations

import argparse
import socket
import sys
import webbrowser
from collections.abc import Sequence
from pathlib import Path

from halley.gui.app import App
from halley.gui.server import GuiServer, make_server

DEFAULT_PORT = 8765
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
WILDCARD_HOSTS = ("0.0.0.0", "::", "")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m halley.gui",
        description="Halley's GUI: build a model, run it, read its runs, explain a value and "
        "compare two runs, in the browser.",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("runs"),
        help="the folder whose run directories the GUI lists and where its runs go "
        "(default: runs, in the current folder)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        action="append",
        default=[],
        metavar="FILE",
        help="a Python file of models to load at start; give it once per file",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"the port to listen on (default: {DEFAULT_PORT}, or a free one if it is taken; "
        "0 for any free port)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="the address to listen on (default: 127.0.0.1, this computer only); 0.0.0.0 "
        "listens on every interface",
    )
    parser.add_argument(
        "--allow-host",
        action="append",
        default=[],
        metavar="NAME",
        help="a further name or address other computers reach this one by, which the server "
        "answers to; give it once per name",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="print the address without opening it"
    )
    return parser.parse_args(argv)


def own_names(host: str) -> tuple[str, ...]:
    """The names another computer may reach this one by, for a server that listens on more than
    the loopback interface: the machine's name, and for one that listens on every interface the
    addresses it resolves to and routes from (the server answers any of its addresses then). A
    page of another site that reaches the server through a name it controls still names that
    name, so answering these does not weaken the server's check on the ``Host`` header."""
    if host in LOOPBACK_HOSTS:
        return ()
    names = {socket.gethostname(), socket.getfqdn()}
    if host in WILDCARD_HOSTS:
        names |= routed_addresses()
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None):
                names.add(str(info[4][0]))
        except OSError:
            pass
    return tuple(sorted(n for n in names if n))


def routed_addresses() -> set[str]:
    """The addresses this computer sends from on its default routes, which another computer on
    the network reaches it by, where the machine's name may resolve to a loopback address only
    (Debian and Ubuntu map it to 127.0.1.1). Connecting a UDP socket sends nothing: it only
    picks the route, to an address kept for documentation."""
    found: set[str] = set()
    for family, target in (
        (socket.AF_INET, ("192.0.2.1", 9)),
        (socket.AF_INET6, ("2001:db8::1", 9)),
    ):
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as probe:
                probe.connect(target)
                address = str(probe.getsockname()[0])
        except OSError:
            continue
        if address not in ("0.0.0.0", "::"):
            found.add(address)
    return found


def bind(app: App, host: str, port: int, extra_hosts: tuple[str, ...] = ()) -> GuiServer:
    """A server for ``app`` at ``host:port``, or at a free port when that port cannot be had: in
    use, or on Windows inside a range the system reserves."""
    try:
        return make_server(app.routes(), host, port, extra_hosts=extra_hosts)
    except OSError as exc:
        if port == 0:
            raise
        reason = exc.strerror or str(exc)
        server = make_server(app.routes(), host, 0, extra_hosts=extra_hosts)
        sys.stderr.write(f"cannot listen on port {port} ({reason}); using port {server.port}\n")
        return server


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    app = App(args.runs)
    for path in args.model:
        app.load_model_file(path)
    extra = (*args.allow_host, *own_names(args.host))
    try:
        server = bind(app, args.host, args.port, extra)
    except OSError as exc:
        sys.stderr.write(f"cannot listen on {args.host}: {exc.strerror or exc}\n")
        app.close()
        return 2
    url = server.browser_url()
    if args.host not in LOOPBACK_HOSTS:
        sys.stderr.write(
            f"listening on {args.host}: anyone who can reach this address and holds the token "
            "can run Python code on this computer through the GUI\n"
        )
    sys.stdout.write(f"Halley GUI at {url}\nRuns folder: {app.runs_root}\nCtrl+C stops it.\n")
    sys.stdout.flush()
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        running = app.jobs.running()
        app.close()
        if running is not None:
            sys.stdout.write(
                f"Waiting for {running.label} to finish: a build or run cannot be stopped part "
                "way. Close this window to end it now.\n"
            )
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
