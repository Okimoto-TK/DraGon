FETCH_FIELD_UNIVERSE = [
    "ts_code",
    "exchange",
    "list_status"
]

FETCH_FIELD_CAL = [
    "cal_date",
    "is_open",
]

FETCH_FIELD_DAILY = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "vol",
]

FETCH_FIELD_ADJ_FACTOR = [
    "ts_code",
    "trade_date",
    "adj_factor"
]

FETCH_FIELD_5MIN = [
    "code",
    "t",
    "o",
    "h",
    "l",
    "c",
    "v"
]

FETCH_FIELD_MONEYFLOW = [
    "ts_code",
    "trade_date",
    "buy_sm_vol",
    "buy_sm_amount",
    "sell_sm_vol",
    "sell_sm_amount",
    "buy_md_vol",
    "buy_md_amount",
    "sell_md_vol",
    "sell_md_amount",
    "buy_lg_vol",
    "buy_lg_amount",
    "sell_lg_vol",
    "sell_lg_amount",
    "buy_elg_vol",
    "buy_elg_amount",
    "sell_elg_vol",
    "sell_elg_amount",
    "net_mf_vol",
    "net_mf_amount"
]

FETCH_FIELD_LIMIT = [
    "trade_date",
    "ts_code",
    "up_limit",
    "down_limit"
]

FETCH_FIELD_NAMECHANGE = [
    "ts_code",
    "name",
    "start_date",
    "ann_date",
]

FETCH_FIELD_SUSPEND = [
    "ts_code",
    "trade_date",
    "suspend_type"
]

FIELD_MAP_UNIVERSE = {
    "ts_code": "code",
    "list_status": "status"
}

FIELD_MAP_CAL = {
    "cal_date": "trade_date",
}

FIELD_MAP_DAILY = {
    "ts_code": "code",
}

FIELD_MAP_ADJ_FACTOR = FIELD_MAP_DAILY

FIELD_MAP_5MIN = {
    "t": "trade_time",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "vol",
}

FIELD_MAP_MONEYFLOW = FIELD_MAP_DAILY

FIELD_MAP_NAMECHANGE = {
    "ts_code": "code",
    "start_date": "trade_date",
}

FIELD_MAP_LIMIT = FIELD_MAP_DAILY

FIELD_MAP_SUSPEND = FIELD_MAP_DAILY
