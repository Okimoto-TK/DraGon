"""Entry point for running the raw data pipeline."""
from __future__ import annotations

from src.data.models import Query
from src.data.pipelines import RawPipeline

# Date range for data fetching
START_DATE = "20260214"
END_DATE = "20260410"


def main() -> None:
    """Run the raw data pipeline."""
    pipeline = RawPipeline()

    # Uncomment below to fetch and load data
    # pipeline.run(
    #     action={"fetch", "validate"},
    #     query=Query(desc="daily", start_date=START_DATE, end_date=END_DATE),
    # )

    # pipeline.run(
    #     action={"fetch", "validate"},
    #     query=Query(desc="adj_factor", start_date=START_DATE, end_date=END_DATE),
    # )

    # pipeline.run(
    #     action={"fetch", "validate"},
    #     query=Query(desc="5min", start_date=START_DATE, end_date=END_DATE),
    # )

    pipeline.run(
        action={"fetch", "validate"},
        query=Query(desc="moneyflow", start_date=START_DATE, end_date=END_DATE),
    )
    #
    # pipeline.run(
    #     action={"fetch", "validate"},
    #     query=Query(desc="limit", start_date=START_DATE, end_date=END_DATE),
    # )
    #
    # pipeline.run(
    #     action={"fetch", "validate"},
    #     query=Query(desc="namechange", start_date=START_DATE, end_date=END_DATE),
    # )
    # pipeline.run(
    #     action={"fetch", "validate"},
    #     query=Query(desc="suspend", start_date=START_DATE, end_date=END_DATE),
    # )


if __name__ == "__main__":
    main()

