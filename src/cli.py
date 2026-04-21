"""Project command-line entrypoints."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from typing import Sequence

from config.data import checkpoint_dir as DEFAULT_CHECKPOINT_ROOT
from src.data.models import Query
from src.data.pipelines import AssembledPipeline, ProcessedPipeline, RawPipeline
from src.data.registry.processed import PROCESSED_PARAM_MAP
from src.data.registry.raw import PARAM_MAP
from src.train import run_training


def _parse_field_list(values: Iterable[str]) -> list[str]:
    fields: list[str] = []
    for value in values:
        parts = [part.strip() for part in value.split(",")]
        fields.extend(part for part in parts if part)
    if not fields:
        raise ValueError("At least one field must be provided.")
    return fields


def _run_prepare(
    *,
    layer: str,
    fields: Sequence[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[str]:
    if layer == "raw":
        pipeline = RawPipeline()
        for field in fields:
            pipeline.run(
                action={"fetch", "validate"},
                query=Query(desc=field, start_date=start_date, end_date=end_date),
            )
        return list(fields)

    if layer == "assembled":
        AssembledPipeline.run()
        return []

    raw_pipe = RawPipeline()
    pipeline = ProcessedPipeline(raw_pipe=raw_pipe)
    for field in fields:
        pipeline.run(
            action={"process", "validate"},
            desc=field,
        )
    return list(fields)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dragon")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the forecast model")
    train_parser.add_argument(
        "-n",
        "--name",
        dest="run_name",
        help="Write outputs to models/checkpoints/<name>. May be combined with --load/--checkpoint to migrate from an existing checkpoint into a new run directory.",
    )
    resume_group = train_parser.add_mutually_exclusive_group(required=False)
    resume_group.add_argument(
        "-l",
        "--load",
        dest="load_name",
        help="Resume from models/checkpoints/<name>/latest.pt. May be combined with --name to resume into a new run directory.",
    )
    resume_group.add_argument(
        "-c",
        "--checkpoint",
        dest="checkpoint",
        help="Resume from an explicit checkpoint file or checkpoint directory. May be combined with --name to resume into a new run directory.",
    )
    train_parser.add_argument(
        "-t",
        "--task",
        choices=("ret", "rv", "q"),
        help="Train a single selected task head: ret, rv, or q.",
    )

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Fetch raw data, build processed features, or assemble packed tensors",
    )
    prepare_parser.add_argument(
        "layer",
        choices=("raw", "processed", "assembled"),
        help="Pipeline layer to prepare",
    )
    prepare_parser.add_argument(
        "fields",
        nargs="*",
        help="Comma-separated or space-separated field names",
    )
    prepare_parser.add_argument(
        "-s",
        "--start",
        dest="start_date",
        help="Start date for raw fetching, format YYYYMMDD",
    )
    prepare_parser.add_argument(
        "-e",
        "--end",
        dest="end_date",
        help="End date for raw fetching, format YYYYMMDD",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        result = run_training(
            name=args.run_name,
            load_name=args.load_name,
            checkpoint=args.checkpoint,
            checkpoint_root=DEFAULT_CHECKPOINT_ROOT,
            task=args.task,
        )
        print(result)
        return

    if args.command == "prepare":
        if args.layer == "raw":
            fields = _parse_field_list(args.fields)
            if (args.start_date is None) ^ (args.end_date is None):
                parser.error("--start and --end must be provided together for raw preparation.")
            invalid = [field for field in fields if field not in PARAM_MAP]
            if invalid:
                parser.error(
                    "Unsupported raw fields: "
                    f"{', '.join(invalid)}. "
                    f"Choices: {', '.join(sorted(PARAM_MAP))}"
                )
        elif args.layer == "processed":
            fields = _parse_field_list(args.fields)
            if args.start_date is not None or args.end_date is not None:
                parser.error("--start/--end are only supported when layer=raw.")
            invalid = [field for field in fields if field not in PROCESSED_PARAM_MAP]
            if invalid:
                parser.error(
                    "Unsupported processed fields: "
                    f"{', '.join(invalid)}. "
                    f"Choices: {', '.join(sorted(PROCESSED_PARAM_MAP))}"
                )
        else:
            fields = []
            if args.fields:
                parser.error("layer=assembled does not accept field arguments.")
            if args.start_date is not None or args.end_date is not None:
                parser.error("--start/--end are only supported when layer=raw.")

        prepared = _run_prepare(
            layer=args.layer,
            fields=fields,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        print({"layer": args.layer, "fields": prepared})


if __name__ == "__main__":
    main()
