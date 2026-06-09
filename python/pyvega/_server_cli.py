"""Launch the Vega C++ session server (wrapper around vega_server binary)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from pyvega._client import find_server_binary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Vega C++ session server.")
    parser.add_argument("--bind", default="127.0.0.1:9500", help="host:port to bind")
    parser.add_argument("--print-listening", action="store_true")
    args = parser.parse_args()

    binary = find_server_binary()
    host, _, port = args.bind.partition(":")
    cmd = [binary, "--bind", host, "--port", port or "9500"]
    if args.print_listening:
        cmd.append("--print-listening")
    os.execv(binary, cmd)


if __name__ == "__main__":
    main()
