#!/usr/bin/env python3
"""Validate a DM-Tac W Tabero-style LeRobot v2.1 export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .raw_dmtac_w_to_tabero_lerobot import validate_tabero_lerobot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Converted LeRobot v2.1 root")
    args = parser.parse_args()
    print(json.dumps(validate_tabero_lerobot(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
