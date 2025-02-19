from service_capacity_modeling.models.headroom_strategy import (
    QueuingBasedHeadroomStrategy,
)


def test_calculate_headroom_using_default_strategy():
    effective_cpu_small_instance = 4
    effective_cpu_large_instance = 16
    headroom_strategy = QueuingBasedHeadroomStrategy()
    reserved_headroom_small_instance = headroom_strategy.calculate_reserved_headroom(
        effective_cpu_small_instance
    )
    reserved_headroom_large_instance = headroom_strategy.calculate_reserved_headroom(
        effective_cpu_large_instance
    )

    # The headroom should be inversely proportional to effective CPU
    assert reserved_headroom_large_instance < reserved_headroom_small_instance
