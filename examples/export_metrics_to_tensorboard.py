"""Export MIP metrics.jsonl and offline eval summaries to TensorBoard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--tb-root", type=Path, default=None)
    return parser.parse_args()


def export_run(run_dir: Path, tb_root: Path | None) -> Path:
    tb_dir = (tb_root or (run_dir / "tensorboard")) / "scalars" / run_dir.name
    writer = SummaryWriter(log_dir=str(tb_dir))

    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                step = int(record.pop("step"))
                for key, value in record.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"train/{key}", value, step)

    for summary_path in sorted(run_dir.glob("offline_eval_*.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        step = int(summary.get("step", 0))
        tag = summary.get("tag", summary_path.stem)
        for key in ("action_mse", "action_mae", "action_max_abs_error", "num_sequences"):
            value = summary.get(key)
            if isinstance(value, (int, float)):
                writer.add_scalar(f"offline_eval/{tag}/{key}", value, step)
        writer.add_text(f"offline_eval/{tag}/summary_json", json.dumps(summary, indent=2), step)

    writer.flush()
    writer.close()
    return tb_dir


def main() -> int:
    args = parse_args()
    for run_dir in args.run_dirs:
        print(export_run(run_dir.resolve(), args.tb_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
