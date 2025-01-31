from service_capacity_modeling.interface import Instance
from service_capacity_modeling.tools.auto_shape import reshape_family


def test_reshape_family():
    result = reshape_family("r6id.2xlarge")

    expected_large = Instance.model_validate(
        {
            "name": "r6id.large",
            "cpu": 2,
            "cpu_ghz": 3.5,
            "ram_gib": 15.45,
            "net_mbps": 781.25,
            "drive": {
                "name": "ephem",
                "size_gib": 110,
                "read_io_latency_ms": {
                    "minimum_value": 0.05,
                    "low": 0.10,
                    "mid": 0.125,
                    "high": 0.17,
                    "maximum_value": 2.00,
                    "confidence": 0.9,
                },
                "read_io_per_s": 33542,
                "write_io_per_s": 16771,
                "block_size_kib": 4,
                "single_tenant": False,
            },
        }
    )
    assert result[0].name == "r6id.large"
    assert result[0] == expected_large

    expected_24xlarge = Instance.model_validate(
        {
            "name": "r6id.24xlarge",
            "cpu": 96,
            "cpu_ghz": 3.5,
            "ram_gib": 741.6,
            "net_mbps": 37500.0,
            "drive": {
                "name": "ephem",
                "size_gib": 5292,
                "read_io_latency_ms": {
                    "minimum_value": 0.05,
                    "low": 0.10,
                    "mid": 0.125,
                    "high": 0.17,
                    "maximum_value": 2.05,
                    "confidence": 0.9,
                },
                "read_io_per_s": 1610004,
                "write_io_per_s": 805008,
                "block_size_kib": 4,
                "single_tenant": True,
            },
        }
    )

    assert result[-1].name == "r6id.24xlarge"
    assert result[-1] == expected_24xlarge


def test_reshape_with_override():
    result = reshape_family("r6id.2xlarge")

    for shape in result:
        assert shape.drive is not None
        if shape.name == "r6id.large":
            # This shape has a hard coded override on the drive
            assert shape.drive.read_io_latency_ms.maximum_value == 2.0
        else:
            assert shape.drive.read_io_latency_ms.maximum_value == 2.05
