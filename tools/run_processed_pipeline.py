"""Entry point for running the processed data pipeline."""
from __future__ import annotations

from src.data.pipelines import RawPipeline, ProcessedPipeline

# Processed feature types to generate
FEATURE_TYPES = [
    # "index",
    # "mask",
    # "macro",
    "mezzo",
    "micro",
    # "sidechain",
    # "label",
]


def main() -> None:
    """Run the processed data pipeline."""
    raw_pipe = RawPipeline()
    processed_pipe = ProcessedPipeline(raw_pipe=raw_pipe)

    for desc in FEATURE_TYPES:
        print(f"Processing {desc}...")
        processed_pipe.run(
            action={"process", "validate"},
            desc=desc,
        )
        print(f"Done processing {desc}.")


if __name__ == "__main__":
    main()

