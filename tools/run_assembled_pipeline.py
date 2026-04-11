"""Entry point for running the processed data pipeline."""
from __future__ import annotations

from src.data.pipelines import AssembledPipeline


def main() -> None:
    """Run the processed data pipeline."""
    pipeline = AssembledPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()

