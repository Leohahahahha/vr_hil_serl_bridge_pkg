from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..tactile_csv_utils import (
    LEROBOT_META_COLUMNS,
    read_json,
    sides_from_arg,
    tactile_value_columns,
    vector_value_dict,
    write_json,
)
from .common import TACTILE_LEFT_KEY, TACTILE_RIGHT_KEY


SIDE_KEYS = {
    "left": TACTILE_LEFT_KEY,
    "right": TACTILE_RIGHT_KEY,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LeRobot tactile columns to wide CSV files.")
    parser.add_argument("--dataset-root", required=True, help="Exported LeRobot v3-style dataset root")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to <dataset-root>/exports/tactile_csv")
    parser.add_argument("--side", choices=["left", "right", "both"], default="both", help="Tactile side to export")
    parser.add_argument("--start-index", type=int, default=0, help="First global dataset row index to export")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of rows to export")
    args = parser.parse_args()

    result = export_lerobot_tactile_csv(
        dataset_root=Path(args.dataset_root),
        out_dir=None if args.out_dir is None else Path(args.out_dir),
        side=args.side,
        start_index=int(args.start_index),
        limit=None if args.limit is None else int(args.limit),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def export_lerobot_tactile_csv(
    *,
    dataset_root: Path,
    out_dir: Path | None,
    side: str = "both",
    start_index: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    out_dir = (out_dir or (dataset_root / "exports" / "tactile_csv")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    info = read_json(dataset_root / "meta" / "info.json")
    df = _read_data(dataset_root)
    total_rows = int(len(df))
    if total_rows <= 0:
        raise ValueError(f"{dataset_root} has no data rows")
    if start_index < 0 or start_index >= total_rows:
        raise IndexError(f"start-index {start_index} out of range [0, {total_rows})")
    end_index = total_rows if limit is None else min(total_rows, start_index + max(0, int(limit)))
    data = df.iloc[start_index:end_index].reset_index(drop=True)

    results = {}
    for side_name in sides_from_arg(side):
        results[side_name] = _export_side(
            dataset_root=dataset_root,
            out_dir=out_dir,
            side=side_name,
            info=info,
            df=data,
            start_index=start_index,
            total_rows=total_rows,
        )

    summary = {
        "dataset_root": str(dataset_root),
        "out_dir": str(out_dir),
        "tactile_profile": info.get("tactile_profile"),
        "sides": results,
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def _read_data(dataset_root: Path) -> pd.DataFrame:
    paths = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet files under {dataset_root / 'data'}")
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _export_side(
    *,
    dataset_root: Path,
    out_dir: Path,
    side: str,
    info: dict[str, Any],
    df: pd.DataFrame,
    start_index: int,
    total_rows: int,
) -> dict[str, Any]:
    key = SIDE_KEYS[side]
    feature = info["features"][key]
    dim = int(feature["shape"][0])
    dtype = str(feature["dtype"])
    columns = tactile_value_columns(
        dim=dim,
        tactile_profile=str(info.get("tactile_profile", "")),
        msg_package=str(info.get("tactile_msg_package", "")),
    )

    rows = []
    for offset, (_, source_row) in enumerate(df.iterrows()):
        values = np.asarray(source_row[key], dtype=np.float32).reshape(-1)
        row = {
            "side": side,
            "global_index": int(source_row["index"]) if "index" in source_row else int(start_index + offset),
            "episode_index": int(source_row["episode_index"]),
            "frame_index": int(source_row["frame_index"]),
            "timestamp": float(source_row["timestamp"]),
            "task_index": int(source_row["task_index"]),
            "dtype": dtype,
        }
        row.update(vector_value_dict(values, columns))
        rows.append(row)

    csv_path = out_dir / f"lerobot_tactile_{side}.csv"
    pd.DataFrame(rows, columns=LEROBOT_META_COLUMNS + columns).to_csv(csv_path, index=False)
    return {
        "csv": str(csv_path),
        "column": key,
        "rows": int(len(rows)),
        "total_dataset_rows": total_rows,
        "start_index": int(start_index),
        "end_index": int(start_index + len(rows)),
        "dim": dim,
        "dtype": dtype,
        "value_columns": columns,
    }


if __name__ == "__main__":
    main()
