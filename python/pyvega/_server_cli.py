"""Run the Vega session server: ``vega-server --bind 127.0.0.1:9500``.

Uses the compiled ``pyvega._vega_server`` extension (shipped in wheels) when
available, running the server inside this interpreter; otherwise execs a
standalone ``vega_server`` binary (``VEGA_SERVER`` or ``PATH``).
"""

from __future__ import annotations

import argparse
import os
import signal
import sys


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="vega-server", description="Run the Vega session server.")
    parser.add_argument("--bind", default="127.0.0.1:9500", help="host[:port] to bind")
    parser.add_argument("--port", type=int, default=None, help="port (overrides --bind's port)")
    parser.add_argument("--print-listening", action="store_true",
                        help="print 'LISTENING <port>' once ready")
    parser.add_argument("--embedded", action="store_true",
                        help="require the in-process extension server")
    parser.add_argument("--binary", action="store_true",
                        help="force the standalone vega_server binary")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse(argv)
    host, _, port_str = args.bind.partition(":")
    port = args.port if args.port is not None else int(port_str or 9500)
    host = host or "127.0.0.1"

    if not args.binary:
        try:
            from pyvega import _vega_server
        except ImportError:
            if args.embedded:
                raise
            _vega_server = None
        if _vega_server is not None:
            def _terminate(signum, frame):  # noqa: ARG001
                raise SystemExit(0)

            signal.signal(signal.SIGTERM, _terminate)
            try:
                _vega_server.serve(host, port, args.print_listening)
            except (KeyboardInterrupt, SystemExit):
                pass
            # Session threads may still hold Python objects; skip interpreter
            # finalization rather than racing them.
            sys.stdout.flush()
            os._exit(0)

    from pyvega._client import find_server_binary

    binary = find_server_binary()
    cmd = [binary, "--bind", host, "--port", str(port)]
    if args.print_listening:
        cmd.append("--print-listening")
    os.execv(binary, cmd)


if __name__ == "__main__":
    main()
