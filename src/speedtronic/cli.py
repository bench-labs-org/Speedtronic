"""Command-line interface for Speedtronic."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .config import ConfigError, SpeedtronicConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="speedtronic", description="Efficient training with optional DumbDiLoCo"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="run or resume a training job")
    train.add_argument("--config", required=True, help="YAML or JSON configuration path")
    train.add_argument("--resume", action="store_true", help="resume the latest local checkpoint")
    train.add_argument("--device", default=None, help="override run.device (auto, cpu, cuda, mps)")
    train.add_argument(
        "--max-steps", type=int, default=None, help="override the total local optimizer-step target"
    )
    train.add_argument("--output-dir", default=None, help="override run.output_dir")

    validate = subparsers.add_parser("validate", help="validate and print a configuration")
    validate.add_argument("--config", required=True, help="YAML or JSON configuration path")
    validate.add_argument("--format", choices=("yaml", "json"), default="yaml")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = SpeedtronicConfig.load(args.config)
        if args.command == "validate":
            rendered = (
                json.dumps(config.to_dict(), indent=2)
                if args.format == "json"
                else config.to_yaml()
            )
            print(rendered)
            return 0
        if args.output_dir:
            config.run.output_dir = args.output_dir
        if args.max_steps is not None:
            config.run.max_steps = args.max_steps
            config.scheduler.max_steps = args.max_steps
        # Import runtime lazily so ``validate`` remains useful in a minimal
        # environment and gives a focused error if torch is not installed.
        from .runtime import train_from_config

        result = train_from_config(
            config,
            resume=bool(args.resume),
            device=args.device,
            max_steps=args.max_steps,
        )
        print(
            f"completed steps={result.steps} samples={result.samples} "
            f"tokens={result.tokens} loss={result.final_loss}"
        )
        return 0
    except (ConfigError, FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(f"speedtronic: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
