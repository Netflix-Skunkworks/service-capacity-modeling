from typing import Optional
from typing import Tuple

import pytest

from service_capacity_modeling.demand import Demand
from service_capacity_modeling.demand import DIMENSIONS
from service_capacity_modeling.demand import fit
from service_capacity_modeling.demand import Fit
from service_capacity_modeling.demand import ncu_per_node
from service_capacity_modeling.demand import NodeRules
from service_capacity_modeling.demand import plan
from service_capacity_modeling.demand import Resource
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import Excuse
from service_capacity_modeling.interface import Interval

HARDWARE = shapes.region("us-east-1")
GP3 = HARDWARE.drives["gp3"]

# One zone needs 140 NCU to serve its load at the CPU target; it can run on
# as little as 120 while a weaker fallback is in the fleet.
DEMAND = Demand(
    ncu=Interval(low=120, mid=140, high=160),
    mem_gib=Interval(low=60, mid=90, high=120),
    disk_gib=Interval(low=600, mid=800, high=1000),
    network_mbps=Interval(low=1000, mid=2000, high=3000),
)
RULES = NodeRules(reserve_mem_gib=lambda _: 4, max_data_per_node_gib=1536, min_count=2)


class FixedDemandModel:
    def demand(
        self, desires: CapacityDesires, drive: Drive
    ) -> Optional[Tuple[Demand, NodeRules]]:
        del desires
        return (DEMAND, RULES) if drive.name == "gp3" else None


def test_ncu_normalizes_cpu_speed():
    assert ncu_per_node(HARDWARE.instances["m6id.2xlarge"]) == pytest.approx(14.0)
    assert ncu_per_node(HARDWARE.instances["m7a.2xlarge"]) == pytest.approx(
        25.1, abs=0.1
    )


def test_count_comes_from_the_binding_resource():
    m7a = fit(DEMAND, RULES, HARDWARE.instances["m7a.2xlarge"], GP3)

    assert isinstance(m7a, Fit)
    assert m7a.count == 6  # ceil(140 / 25.1); memory needs only ceil(90 / 26.5)
    assert m7a.binding == Resource.ncu


def test_alternative_must_meet_the_floor_at_the_primary_count():
    weaker = fit(DEMAND, RULES, HARDWARE.instances["m6id.2xlarge"], GP3, count=6)
    stronger = fit(DEMAND, RULES, HARDWARE.instances["r7a.2xlarge"], GP3, count=6)

    assert isinstance(weaker, Excuse)  # 6 x 14 NCU is below the 120 floor
    assert weaker.context["supply_per_node"] == pytest.approx(14.0)
    assert isinstance(stronger, Fit)


def test_plan_lists_alternatives_that_can_replace_a_primary_node():
    names = ["m6id.2xlarge", "m6a.2xlarge", "m7a.2xlarge", "r7a.2xlarge", "m7i.2xlarge"]
    result = plan(
        FixedDemandModel(),
        CapacityDesires(),
        [HARDWARE.instances[n] for n in names],
        [GP3],
    )

    assert result is not None
    floor = DEMAND.at("low")
    for alt in result.alternatives:
        assert alt.count == result.primary.count
        assert alt.instance.family != result.primary.instance.family
        for dim in DIMENSIONS:
            assert alt.supply_per_node[dim] * alt.count >= floor[dim]
