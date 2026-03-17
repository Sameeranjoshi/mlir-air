#!/usr/bin/env python3
"""Export AIR .dot dependency graphs to Model Explorer JSON files.

The built-in JSON adapter (file loader) only accepts these shapes (see ``gk`` / ``Z0e``
in the Model Explorer web bundle):

1. ``{"label": "<name>", "graphs": [ {...}, ... ]}``  — graph collection
2. ``[ { "id": "...", "nodes": [...] }, ... ]`` — bare list of graphs

It does *not* accept ``{"graphs": [...]}`` without ``label``; that yields
``Unsupported JSON format`` and shows Error in the UI.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from model_explorer import graph_builder
from model_explorer.utils import remove_none

from .dot_parser import parse_air_dot


def _strip_empty_strings(obj):
    """Remove dict keys whose values are empty strings (helps WebGL/style parsers)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            cv = _strip_empty_strings(v)
            if cv == "":
                continue
            out[k] = cv
        return out
    if isinstance(obj, list):
        return [_strip_empty_strings(x) for x in obj]
    return obj


def _apply_flatten_namespaces(graphs):
    for g in graphs:
        for n in g.nodes:
            n.namespace = ""


def _apply_rankdir_lr(graphs):
    """Match AIR DOT rankdir=LR for group layers."""
    cfg = graph_builder.GroupNodeConfig(
        namespaceRegex=".*",
        layoutDirection=graph_builder.LayoutDirection.LEFT_RIGHT,
    )
    for g in graphs:
        g.groupNodeConfigs = [cfg]


def graphs_to_json_payload(
    graphs,
    label: str,
    *,
    flatten_namespaces: bool = False,
    rankdir_lr: bool = False,
) -> dict:
    """Return a dict that Model Explorer's JSON file loader accepts."""
    if flatten_namespaces:
        _apply_flatten_namespaces(graphs)
    if rankdir_lr:
        _apply_rankdir_lr(graphs)
    cleaned = [
        _strip_empty_strings(remove_none(asdict(g)))
        for g in graphs
    ]
    return {
        "label": label,
        "graphs": cleaned,
    }


def export_dot_to_json(
    dot_path: str,
    out_path: str | None = None,
    label: str | None = None,
    *,
    flatten_namespaces: bool = False,
    rankdir_lr: bool = False,
) -> str:
    dot_p = Path(dot_path)
    graphs = parse_air_dot(str(dot_p))
    lbl = label or dot_p.stem
    payload = graphs_to_json_payload(
        graphs,
        lbl,
        flatten_namespaces=flatten_namespaces,
        rankdir_lr=rankdir_lr,
    )
    out = Path(out_path) if out_path else dot_p.with_suffix(".model_explorer.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return str(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert AIR .dot to Model Explorer JSON (label + graphs)."
    )
    parser.add_argument("dot_path", help="Input .dot file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output .json path (default: <dot_stem>.model_explorer.json)",
    )
    parser.add_argument(
        "--label",
        help="Collection label in Model Explorer (default: stem of input file)",
    )
    parser.add_argument(
        "--flatten-namespaces",
        action="store_true",
        help="Set every node's namespace to empty so all ops show at root (easier to see layers).",
    )
    parser.add_argument(
        "--rankdir-lr",
        action="store_true",
        help="Use left-right layout for namespace groups (closer to Graphviz rankdir=LR).",
    )
    args = parser.parse_args()
    path = export_dot_to_json(
        args.dot_path,
        args.output,
        args.label,
        flatten_namespaces=args.flatten_namespaces,
        rankdir_lr=args.rankdir_lr,
    )
    print(path)


if __name__ == "__main__":
    main()
