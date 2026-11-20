"""Shared command line: ``--config`` (default configs/pilot.yaml, the ST plant) and ``--out``
(default: the config's ``out`` key, e.g. results/ for ST and results_std/ for configs/pilot_std.yaml)."""

from __future__ import annotations

import argparse
import pathlib

import yaml


def parse(description: str = ""):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = pathlib.Path(args.out or cfg.get("out", "results"))
    out.mkdir(parents=True, exist_ok=True)
    return cfg, args.config, out
