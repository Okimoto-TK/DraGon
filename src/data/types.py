from typing import Dict, Sequence, Optional
from pydantic import BaseModel, model_validator, ConfigDict

Map = Dict[str, str]


class Query(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    desc: str
    trade_date: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    codes: Optional[Sequence[str]] = None

    @model_validator(mode="after")
    def check(self):
        if self.trade_date is not None and (self.start_date is not None or self.end_date is not None):
            raise ValueError("trade_date cannot be used together with start_date/end_date")

        if (self.start_date is None) ^ (self.end_date is None):
            raise ValueError("start_date and end_date must be provided together")

        if self.codes is not None:
            if isinstance(self.codes, str):
                raise TypeError("codes must be a sequence of strings, not a single string")
            for code in self.codes:
                if not isinstance(code, str):
                    raise TypeError("all items in codes must be strings")

        return self
        