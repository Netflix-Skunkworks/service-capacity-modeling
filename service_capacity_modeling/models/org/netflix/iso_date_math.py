import math
from datetime import timedelta
from decimal import Decimal
from typing import Union

import isodate  # type: ignore
from isodate import parse_duration

DURATION_1M = 2592000
DURATION_1Y = 31536000
D_YEAR = Decimal(DURATION_1Y)
D_MONTH = Decimal(DURATION_1M)


def _iso_to_proto_duration(iso_duration: str) -> str:
    parsed = parse_duration(iso_duration)
    return f"{int(parsed.total_seconds())}s"


def iso_to_seconds(iso_duration: str, unlimited: int = 0) -> int:
    if iso_duration == "unlimited":
        return unlimited
    parsed: Union[timedelta, isodate.Duration] = isodate.parse_duration(iso_duration)
    # isodate package parse_duration returns either timedelta OR Duration
    if isinstance(parsed, isodate.Duration):
        return int(
            math.ceil(
                parsed.years * D_YEAR
                + parsed.months * D_MONTH
                + Decimal(parsed.tdelta.total_seconds())
            )
        )
    return int(parsed.total_seconds())


def _iso_to_timedelta(iso_duration: str, unlimited: int = 0) -> timedelta:
    return timedelta(seconds=iso_to_seconds(iso_duration, unlimited))
