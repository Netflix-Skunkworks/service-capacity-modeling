from datetime import timedelta

from service_capacity_modeling.models.org.netflix.time_series_config import (
    TimeSeriesConfiguration,
)


def test_timeseries_simple_config_derivation():
    config = TimeSeriesConfiguration(
        {
            "ts.read-interval": "PT10H",
            "ts.hot.retention-interval": "PT8640H",
            "ts.fire-and-forget": False,
            "ts.events-per-day-per-ts": "10",
            "ts.event-size": "10000",
        }
    )

    assert config.accept_limit == "600s"
    assert config.seconds_per_slice == int(timedelta(days=180).total_seconds())
    assert config.seconds_per_interval == 36000
    assert config.buckets_per_id == 1
    assert config.seconds_per_bucket == 36000
    assert config.coalesce_duration == "0.001s"
    assert config.read_amplification == 1


def test_timeseries_read_amplified_config_derivation():
    config = TimeSeriesConfiguration(
        {
            "ts.read-interval": "PT24H",
            "ts.hot.retention-interval": "PT96H",
            "ts.fire-and-forget": True,
            "ts.events-per-day-per-ts": "1000",
            "ts.event-size": "20000",
        }
    )

    assert config.accept_limit == "600s"
    assert config.seconds_per_slice == int(timedelta(days=1).total_seconds())
    assert config.seconds_per_interval == int(timedelta(days=1).total_seconds())
    assert config.buckets_per_id == 5
    assert config.seconds_per_bucket == 5
    assert config.coalesce_duration == "1s"
    assert config.read_amplification == 5
