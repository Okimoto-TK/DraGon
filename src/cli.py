"""Project command-line entrypoints."""
from __future__ import annotations

import argparse
from typing import Sequence

from config.config import checkpoint_dir as DEFAULT_CHECKPOINT_DIR
from config.config import mlflow_dir as DEFAULT_MLFLOW_DIR
from src.train.fit import run_training


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dragon")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the multiscale fusion model")
    train_parser.add_argument("name", help="Run name used for MLflow metadata and checkpoint folder")
    train_parser.add_argument("--load", dest="load_checkpoint", help="Checkpoint path or checkpoint directory to resume from")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        result = run_training(
            run_name=args.name,
            load_checkpoint=args.load_checkpoint,
            checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
            mlflow_repo=DEFAULT_MLFLOW_DIR,
        )
        print(result)


if __name__ == "__main__":
    main()
