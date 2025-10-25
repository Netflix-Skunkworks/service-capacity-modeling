import math
from datetime import timedelta
from typing import Any
from typing import Dict

from service_capacity_modeling.models.org.netflix.iso_date_math import (
    _iso_to_proto_duration,
)
from service_capacity_modeling.models.org.netflix.iso_date_math import _iso_to_timedelta
from service_capacity_modeling.models.org.netflix.iso_date_math import iso_to_seconds

DURATION_1H = timedelta(hours=1)
DURATION_4H = timedelta(hours=4)
DURATION_6H = timedelta(hours=6)
DURATION_1D = timedelta(days=1)
DURATION_7D = timedelta(days=7)
DURATION_14D = timedelta(days=14)
DURATION_1M = timedelta(days=30)
DURATION_3M = DURATION_1M * 3
DURATION_6M = DURATION_1M * 6
DURATION_9M = DURATION_1M * 9
DURATION_1Y = timedelta(days=365)
_4MiB = 4 * 1024 * 1024


class TimeSeriesConfiguration:
    def __init__(self, extra_model_arguments: Dict[str, Any]):
        self.read_interval_seconds = self.__get_read_interval_seconds(
            extra_model_arguments
        )
        self.retention_duration = str(
            extra_model_arguments.get("ts.hot.retention-interval", "unlimited")
        )
        self.seconds_per_slice = int(
            self.__get_slice_interval(self.retention_duration).total_seconds()
        )
        self.seconds_per_interval = self.__get_time_interval(
            self.seconds_per_slice, self.read_interval_seconds
        )
        self.buckets_per_id = self.__get_buckets_per_id(
            extra_model_arguments, self.seconds_per_interval
        )
        self.seconds_per_bucket = self.__get_seconds_per_bucket(
            self.read_interval_seconds, self.buckets_per_id, self.seconds_per_interval
        )
        self.coalesce_duration = self.__get_coalesce_duration(extra_model_arguments)
        self.consistency_target = self.__get_consistency_target(extra_model_arguments)
        self.accept_limit = self.__get_accept_limit(
            extra_model_arguments, self.retention_duration
        )
        self.read_amplification = self.__get_read_amplification(
            self.read_interval_seconds, self.seconds_per_interval, self.buckets_per_id
        )
        self.search_enabled = bool(extra_model_arguments.get("search.enabled"))

    # We force dependent parameters to be passed into these methods
    # instead of using self to make dependencies between them more explicit
    @staticmethod
    def __get_accept_limit(
        extra_model_arguments: Dict[str, Any], retention_duration: str
    ) -> str:
        if "ts.accept-limit" in extra_model_arguments:
            return str(_iso_to_proto_duration(extra_model_arguments["ts.accept-limit"]))
        elif retention_duration == "unlimited":
            # For "all-time" tables we have to let them read and write anywhere
            return "0s"
        else:
            return str(_iso_to_proto_duration("PT10M"))

    @staticmethod
    def __get_buckets_per_id(
        extra_model_arguments: Dict[str, Any], seconds_per_interval: int
    ) -> int:
        events_per_day_per_ts = int(
            extra_model_arguments.get("ts.events-per-day-per-ts", "10")
        )
        event_size = int(extra_model_arguments.get("ts.event-size", "1024"))
        bytes_per_interval = (
            seconds_per_interval * events_per_day_per_ts * event_size
        ) / 86400
        return round(max(min(8, bytes_per_interval / _4MiB), 1))

    @staticmethod
    def __get_time_interval(seconds_per_slice: int, read_interval_seconds: int) -> int:
        if seconds_per_slice != 0 and read_interval_seconds > seconds_per_slice:
            # default bucket interval to slice
            # interval to avoid compaction issue seen in C*
            return seconds_per_slice
        return read_interval_seconds

    @staticmethod
    def __get_seconds_per_bucket(
        read_interval_seconds: int, buckets: int, seconds_per_interval: int
    ) -> int:
        if buckets == 1:
            return seconds_per_interval
        # For smaller read intervals, default to 60. Else rotate buckets every 5 secs
        if read_interval_seconds <= 60:
            return 60
        return 5

    @staticmethod
    def __get_slice_interval(retention: str) -> timedelta:
        """
        Choice of slice interval tries to balance the following:
        - Minimal disk waste - the closer the slice interval is to the retention time,
        the more waste there will be
        - Minimal read amplification - when read interval is larger than slice interval,
        there will be a read for each slice within that interval
        - Reasonably frequent slice creations - if slices are created too infrequently,
        it becomes very slow to update slice configuration
        - Intuitive Slice Sizes - we should create slices at regular intervals to
        improve debugging
        """
        # Disable too many returns - the alternate
        # version of this with <6 returns is more complex
        # pylint: disable=too-many-return-statements
        if retention == "unlimited":
            return timedelta(seconds=0)
        retention_seconds = _iso_to_timedelta(retention)
        if retention_seconds < DURATION_6H:
            raise ValueError("Data retention should be at least 6 hours")
        # 6h - 1d retention - 4h slice
        if retention_seconds <= DURATION_1D:
            return DURATION_4H
        # 1d - 7d retention - 1d slice
        if retention_seconds <= DURATION_7D:
            return DURATION_1D
        # 7d - 14d retention - 7d slice
        if retention_seconds <= DURATION_14D:
            return DURATION_7D
        # 14d - 1m retention - 14d slice
        if retention_seconds <= DURATION_1M:
            return DURATION_14D
        # 1m - 3m retention - 1m slice
        if retention_seconds <= DURATION_3M:
            return DURATION_1M
        # 3m - 6m retention - 3m slice
        if retention_seconds <= DURATION_6M:
            return DURATION_3M
        # 6m - 1y retention - 6m slice
        if retention_seconds <= DURATION_1Y:
            return DURATION_6M
        # by default, return 1y slice
        return DURATION_1Y

    @staticmethod
    def __get_read_interval_seconds(extra_model_arguments: Dict[str, Any]) -> int:
        # default read duration of 1D
        read_interval = extra_model_arguments.get("ts.read-interval", "PT24H")
        if read_interval == "unlimited":
            return int(DURATION_1Y.total_seconds() * 100)
        return iso_to_seconds(read_interval)

    @staticmethod
    def __get_coalesce_duration(extra_model_arguments: Dict[str, Any]) -> str:
        # arbitrary defaults. Can be tuned later.
        fire_forget = extra_model_arguments.get("ts.fire-and-forget", False)
        if fire_forget:
            return "1s"
        return "0.001s"

    @staticmethod
    def __get_consistency_target(extra_model_arguments: Dict[str, Any]) -> str:
        val = str(extra_model_arguments.get("ts.read-consistency", "eventual")).lower()
        if val in ("best-effort", "eventual"):
            return val
        else:
            # Default to read-your-writes consistency
            return "read-your-writes"

    @staticmethod
    def __get_read_amplification(
        read_interval_seconds: int, seconds_per_interval: int, buckets_per_id: int
    ) -> int:
        return math.ceil(read_interval_seconds / seconds_per_interval) * buckets_per_id
