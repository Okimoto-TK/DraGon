from typing import Dict, Literal


Map = Dict[str, str]

Exchange = Literal["SSE", "SZSE", "BSE"]
Status = Literal['L', 'D', 'P', 'G']

Action = Literal["fetch", "load", "validate"]
