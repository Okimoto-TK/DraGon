"""Processed pipeline parameter registry."""
from __future__ import annotations

from config.config import processed_path

from src.data.models import ProcessedParams
from src.data.processor.process import (
    process_index,
    process_label,
    process_macro,
    process_mask,
    process_mezzo,
    process_micro,
    process_sidechain,
)
from src.data.registry.processor import MACRO_LOOKBACK, MEZZO_LOOKBACK, MICRO_LOOKBACK
from src.data.schemas.processed import (
    PROCESSED_INDEX_SCHEMA,
    PROCESSED_LABEL_SCHEMA,
    PROCESSED_MACRO_SCHEMA,
    PROCESSED_MASK_SCHEMA,
    PROCESSED_MEZZO_SCHEMA,
    PROCESSED_MICRO_SCHEMA,
    PROCESSED_SIDECHAIN_SCHEMA,
)
from src.data.storage.parquet_io import (
    read_parquet,
    read_parquet_schema,
    read_parquets,
    read_parquets_schema,
    write_parquet,
    write_parquets,
)

# Processed pipeline registry mapping feature type to schema and path
PROCESSED_PARAM_MAP: dict[str, ProcessedParams] = {
    "index": ProcessedParams(
        processor=process_index,
        proc="_process",  # String method name to avoid circular imports
        reader=read_parquet,
        writer=write_parquet,
        sreader=read_parquet_schema,
        path=processed_path.index_path,
        schema=PROCESSED_INDEX_SCHEMA,
        desc="index",
        raw_deps={"suspend_df": "suspend"},
        processor_kwargs={},
    ),
    "mask": ProcessedParams(
        processor=process_mask,
        proc="_process",  # String method name
        reader=read_parquets,
        writer=write_parquets,
        sreader=read_parquets_schema,
        path=processed_path.mask_dir,
        schema=PROCESSED_MASK_SCHEMA,
        desc="mask",
        raw_deps={"suspend_df": "suspend", "namechange_df": "namechange"},
        processor_kwargs={},
    ),
    "macro": ProcessedParams(
        processor=process_macro,
        proc="_process",  # String method name
        reader=read_parquets,
        writer=write_parquets,
        sreader=read_parquets_schema,
        path=processed_path.macro_dir,
        schema=PROCESSED_MACRO_SCHEMA,
        desc="macro",
        raw_deps={"daily_df": "daily", "adj_factor_df": "adj_factor"},
        processor_kwargs={"lookback": MACRO_LOOKBACK},
    ),
    "mezzo": ProcessedParams(
        processor=process_mezzo,
        proc="_process_chunk",  # String method name for chunked processing
        reader=read_parquets,
        writer=write_parquets,
        sreader=read_parquets_schema,
        path=processed_path.mezzo_dir,
        schema=PROCESSED_MEZZO_SCHEMA,
        desc="mezzo",
        raw_deps={},
        processor_kwargs={"lookback": MEZZO_LOOKBACK},
    ),
    "micro": ProcessedParams(
        processor=process_micro,
        proc="_process_chunk",  # String method name for chunked processing
        reader=read_parquets,
        writer=write_parquets,
        sreader=read_parquets_schema,
        path=processed_path.micro_dir,
        schema=PROCESSED_MICRO_SCHEMA,
        desc="micro",
        raw_deps={},
        processor_kwargs={"lookback": MICRO_LOOKBACK},
    ),
    "sidechain": ProcessedParams(
        processor=process_sidechain,
        proc="_process",  # String method name
        reader=read_parquets,
        writer=write_parquets,
        sreader=read_parquets_schema,
        path=processed_path.sidechain_dir,
        schema=PROCESSED_SIDECHAIN_SCHEMA,
        desc="sidechain",
        raw_deps={"daily_df": "daily", "adj_factor_df": "adj_factor", "moneyflow_df": "moneyflow"},
        processor_kwargs={"lookback": MACRO_LOOKBACK},
    ),
    "label": ProcessedParams(
        processor=process_label,
        proc="_process",  # String method name
        reader=read_parquets,
        writer=write_parquets,
        sreader=read_parquets_schema,
        path=processed_path.label_dir,
        schema=PROCESSED_LABEL_SCHEMA,
        desc="label",
        raw_deps={"daily_df": "daily", "adj_factor_df": "adj_factor"},
        processor_kwargs={},
    ),
}
