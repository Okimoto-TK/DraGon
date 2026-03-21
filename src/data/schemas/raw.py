from dataclasses import dataclass
from typing import Literal

DType = Literal["str", "float", "int", "bool"]


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    dtype: DType
    required: bool = True
    nullable: bool = True
    fmt: str | None = None
    unit: str | None = None
    description: str = ""


@dataclass(frozen=True)
class TableSchema:
    name: str
    layer: str
    description: str
    primary_key: tuple[str, ...]
    partition_by: tuple[str, ...]
    columns: tuple[ColumnSchema, ...]
    allow_extra_columns: bool = True
    provider_select_only_schema_columns: bool = True

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(col.name for col in self.columns)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(col.name for col in self.columns if col.required)

    def get_column(self, name: str) -> ColumnSchema:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(f"{name!r} not found in schema {self.name}")


RAW_DAILY_SCHEMA = TableSchema(
    name="raw_daily",
    layer="raw",
    description="Raw daily stock price table.",
    primary_key=("code", "trade_date"),
    partition_by=("trade_date",),
    columns=(
        ColumnSchema(
            name="code",
            dtype="str",
            required=True,
            nullable=False,
            description="The unique code for every stock.",
        ),
        ColumnSchema(
            name="trade_date",
            dtype="str",
            required=True,
            nullable=False,
            fmt="%Y%m%d",
            description="As-of date.",
        ),
        ColumnSchema(
            name="open",
            dtype="float",
            description="Open price.",
            unit="price",
        ),
        ColumnSchema(
            name="close",
            dtype="float",
            description="Close price.",
            unit="price",
        ),
        ColumnSchema(
            name="high",
            dtype="float",
            description="The highest price occurred that day.",
            unit="price",
        ),
        ColumnSchema(
            name="low",
            dtype="float",
            description="The lowest price occurred that day.",
            unit="price",
        ),
        ColumnSchema(
            name="vol",
            dtype="float",
            description="The volume, in per share, not hand.",
            unit="share",
        ),
        ColumnSchema(
            name="adj_factor",
            dtype="float",
            description="Adjustment factor.",
            unit="factor",
        ),
    ),
)


RAW_5MIN_SCHEMA = TableSchema(
    name="raw_5min",
    layer="raw",
    description="Raw 5-minute stock price table.",
    primary_key=("code", "trade_date", "time"),
    partition_by=("trade_date",),
    columns=(
        ColumnSchema(
            name="code",
            dtype="str",
            required=True,
            nullable=False,
            description="The unique code for every stock.",
        ),
        ColumnSchema(
            name="trade_date",
            dtype="str",
            required=True,
            nullable=False,
            fmt="%Y%m%d",
            description="As-of date.",
        ),
        ColumnSchema(
            name="time",
            dtype="str",
            required=True,
            nullable=False,
            fmt="%H:%M:%S",
            description="The specific time of trade.",
        ),
        ColumnSchema(
            name="open",
            dtype="float",
            description="Open price.",
            unit="price",
        ),
        ColumnSchema(
            name="close",
            dtype="float",
            description="Close price.",
            unit="price",
        ),
        ColumnSchema(
            name="high",
            dtype="float",
            description="The highest price occurred in that bar.",
            unit="price",
        ),
        ColumnSchema(
            name="low",
            dtype="float",
            description="The lowest price occurred in that bar.",
            unit="price",
        ),
        ColumnSchema(
            name="vol",
            dtype="float",
            description="The volume, in per share, not hand.",
            unit="share",
        ),
    ),
)


RAW_MONEYFLOW_SCHEMA = TableSchema(
    name="raw_moneyflow",
    layer="raw",
    description="Raw daily money flow table.",
    primary_key=("code", "trade_date"),
    partition_by=("trade_date",),
    columns=(
        ColumnSchema(
            name="code",
            dtype="str",
            required=True,
            nullable=False,
            description="The unique code for every stock.",
        ),
        ColumnSchema(
            name="trade_date",
            dtype="str",
            required=True,
            nullable=False,
            fmt="%Y%m%d",
            description="As-of date.",
        ),
        ColumnSchema(name="buy_sm_vol", dtype="float", description="Volume of small buying orders.", unit="volume"),
        ColumnSchema(name="buy_sm_amount", dtype="float", description="Amount of small buying orders.", unit="currency"),
        ColumnSchema(name="sell_sm_vol", dtype="float", description="Volume of small selling orders.", unit="volume"),
        ColumnSchema(name="sell_sm_amount", dtype="float", description="Amount of small selling orders.", unit="currency"),
        ColumnSchema(name="buy_md_vol", dtype="float", description="Volume of medium buying orders.", unit="volume"),
        ColumnSchema(name="buy_md_amount", dtype="float", description="Amount of medium buying orders.", unit="currency"),
        ColumnSchema(name="sell_md_vol", dtype="float", description="Volume of medium selling orders.", unit="volume"),
        ColumnSchema(name="sell_md_amount", dtype="float", description="Amount of medium selling orders.", unit="currency"),
        ColumnSchema(name="buy_lg_vol", dtype="float", description="Volume of large buying orders.", unit="volume"),
        ColumnSchema(name="buy_lg_amount", dtype="float", description="Amount of large buying orders.", unit="currency"),
        ColumnSchema(name="sell_lg_vol", dtype="float", description="Volume of large selling orders.", unit="volume"),
        ColumnSchema(name="sell_lg_amount", dtype="float", description="Amount of large selling orders.", unit="currency"),
        ColumnSchema(name="buy_elg_vol", dtype="float", description="Volume of extra-large buying orders.", unit="volume"),
        ColumnSchema(name="buy_elg_amount", dtype="float", description="Amount of extra-large buying orders.", unit="currency"),
        ColumnSchema(name="sell_elg_vol", dtype="float", description="Volume of extra-large selling orders.", unit="volume"),
        ColumnSchema(name="sell_elg_amount", dtype="float", description="Amount of extra-large selling orders.", unit="currency"),
        ColumnSchema(name="net_mf_vol", dtype="float", description="Net volume of money flow.", unit="volume"),
        ColumnSchema(name="net_mf_amount", dtype="float", description="Net amount of money flow.", unit="currency"),
    ),
)


RAW_LIMIT_SCHEMA = TableSchema(
    name="raw_limit",
    layer="raw",
    description="Raw daily limit price table.",
    primary_key=("code", "trade_date"),
    partition_by=("trade_date",),
    columns=(
        ColumnSchema(
            name="code",
            dtype="str",
            required=True,
            nullable=False,
            description="The unique code for every stock.",
        ),
        ColumnSchema(
            name="trade_date",
            dtype="str",
            required=True,
            nullable=False,
            fmt="%Y%m%d",
            description="As-of date.",
        ),
        ColumnSchema(name="up_limit", dtype="float", description="The upper limit of price that day.", unit="price"),
        ColumnSchema(name="down_limit", dtype="float", description="The lower limit of price that day.", unit="price"),
    ),
)


RAW_ST_SCHEMA = TableSchema(
    name="raw_st",
    layer="raw",
    description="Raw daily ST status table.",
    primary_key=("code", "trade_date"),
    partition_by=("trade_date",),
    columns=(
        ColumnSchema(
            name="code",
            dtype="str",
            required=True,
            nullable=False,
            description="The unique code for every stock.",
        ),
        ColumnSchema(
            name="trade_date",
            dtype="str",
            required=True,
            nullable=False,
            fmt="%Y%m%d",
            description="As-of date.",
        ),
        ColumnSchema(
            name="is_st",
            dtype="bool",
            required=True,
            nullable=False,
            description="Whether the stock is an ST or *ST stock that day.",
        ),
    ),
)


RAW_SUSPEND_SCHEMA = TableSchema(
    name="raw_suspend",
    layer="raw",
    description="Raw daily suspension status table.",
    primary_key=("code", "trade_date"),
    partition_by=("trade_date",),
    columns=(
        ColumnSchema(
            name="code",
            dtype="str",
            required=True,
            nullable=False,
            description="The unique code for every stock.",
        ),
        ColumnSchema(
            name="trade_date",
            dtype="str",
            required=True,
            nullable=False,
            fmt="%Y%m%d",
            description="As-of date.",
        ),
        ColumnSchema(
            name="is_suspend",
            dtype="bool",
            required=True,
            nullable=False,
            description="Whether the stock is suspended that day.",
        ),
    ),
)


RAW_SCHEMAS = {
    RAW_DAILY_SCHEMA.name: RAW_DAILY_SCHEMA,
    RAW_5MIN_SCHEMA.name: RAW_5MIN_SCHEMA,
    RAW_MONEYFLOW_SCHEMA.name: RAW_MONEYFLOW_SCHEMA,
    RAW_LIMIT_SCHEMA.name: RAW_LIMIT_SCHEMA,
    RAW_ST_SCHEMA.name: RAW_ST_SCHEMA,
    RAW_SUSPEND_SCHEMA.name: RAW_SUSPEND_SCHEMA,
}
