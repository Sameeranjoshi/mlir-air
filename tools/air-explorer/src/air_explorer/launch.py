#!/usr/bin/env python3
"""Launch Model Explorer with an explicit AIR adapter binding."""

from __future__ import annotations

import argparse

from model_explorer import config, visualize_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Launch Model Explorer for AIR graphs with adapterId fixed to "
            "'air_explorer'."
        )
    )
    parser.add_argument("model_path", help="Path to .dot graph file")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument(
        "--no-open-in-browser",
        action="store_true",
        help="Do not auto-open browser tab",
    )
    args = parser.parse_args()

    cfg = config()
    cfg.add_model_from_path(path=args.model_path, adapterId="air_explorer")
    visualize_from_config(
        config=cfg,
        host=args.host,
        port=args.port,
        extensions=["air_explorer"],
        no_open_in_browser=args.no_open_in_browser,
    )


if __name__ == "__main__":
    main()
