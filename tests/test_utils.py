from decimal import Decimal
from typing import Dict
from typing import List

from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.interface import ZoneClusterCapacity
from service_capacity_modeling.models.utils import reduce_by_family


# Create mock hardware instances with different families for all tests
def get_test_instances():
    shape_family_a1 = Instance(
        name="family_a.a1",
        family_separator=".",
        cpu=2,
        cpu_ghz=2.4,
        ram_gib=8,
        net_mbps=1000,
    )
    shape_family_a2 = Instance(
        name="family_a.a2",
        family_separator=".",
        cpu=4,
        cpu_ghz=2.4,
        ram_gib=16,
        net_mbps=2000,
    )
    shape_family_a3 = Instance(
        name="family_a.a3",
        family_separator=".",
        cpu=8,
        cpu_ghz=2.4,
        ram_gib=32,
        net_mbps=4000,
    )

    shape_family_b1 = Instance(
        name="family_b.b1",
        family_separator=".",
        cpu=2,
        cpu_ghz=2.4,
        ram_gib=8,
        net_mbps=1000,
    )
    shape_family_b2 = Instance(
        name="family_b.b2",
        family_separator=".",
        cpu=4,
        cpu_ghz=2.4,
        ram_gib=16,
        net_mbps=2000,
    )

    return (
        shape_family_a1,
        shape_family_a2,
        shape_family_a3,
        shape_family_b1,
        shape_family_b2,
    )


def create_test_capacity_plans() -> List[CapacityPlan]:
    """Create test capacity plans with different hardware families for testing."""
    shapes = get_test_instances()
    (
        shape_family_a1,
        shape_family_a2,
        shape_family_a3,
        shape_family_b1,
        shape_family_b2,
    ) = shapes

    plans = []

    # Family A plans
    for i, shape in enumerate([shape_family_a1, shape_family_a2, shape_family_a3]):
        annual_cost = (i + 1) * 100.0  # Different costs

        cluster = ZoneClusterCapacity(
            cluster_type="test_cluster",
            count=i + 1,
            instance=shape,
            annual_cost=annual_cost,
        )

        annual_costs_a: Dict[str, Decimal] = {"test_cluster": Decimal(str(annual_cost))}

        plans.append(
            CapacityPlan(
                requirements=Requirements(),
                candidate_clusters=Clusters(
                    annual_costs=annual_costs_a, zonal=[cluster], regional=[]
                ),
                cost=annual_cost,
                efficiency=1.0,
            )
        )

    # Family B plans
    for i, shape in enumerate([shape_family_b1, shape_family_b2]):
        annual_cost = (i + 1) * 200.0  # Different costs

        cluster = ZoneClusterCapacity(
            cluster_type="test_cluster",
            count=i + 1,
            instance=shape,
            annual_cost=annual_cost,
        )

        annual_costs_b: Dict[str, Decimal] = {"test_cluster": Decimal(str(annual_cost))}

        plans.append(
            CapacityPlan(
                requirements=Requirements(),
                candidate_clusters=Clusters(
                    annual_costs=annual_costs_b, zonal=[cluster], regional=[]
                ),
                cost=annual_cost,
                efficiency=1.0,
            )
        )

    return plans


def test_reduce_by_family_default():
    """Test that reduce_by_family with default parameter returns one plan per family."""
    plans = create_test_capacity_plans()
    result = reduce_by_family(plans)

    # Should return only 2 plans - one from family_a and one from family_b
    assert len(result) == 2

    # Verify we have one from each family
    families = set()
    for plan in result:
        for cluster in plan.candidate_clusters.zonal:
            families.add(cluster.instance.family)

    assert families == {"family_a", "family_b"}


def test_reduce_by_family_multiple():
    """Test that reduce_by_family with max_results_per_family > 1
    returns multiple plans per family."""
    plans = create_test_capacity_plans()
    result = reduce_by_family(plans, max_results_per_family=2)

    # Should return 4 plans - two from family_a and two from family_b
    assert len(result) == 4

    # Count plans per family
    family_counts = {"family_a": 0, "family_b": 0}
    for plan in result:
        for cluster in plan.candidate_clusters.zonal:
            family_counts[cluster.instance.family] += 1

    # Verify we have exactly 2 from each family
    assert family_counts["family_a"] == 2
    assert family_counts["family_b"] == 2


def test_reduce_by_family_unlimited():
    """Test that reduce_by_family with max_results_per_family > available plans
    returns all plans."""
    plans = create_test_capacity_plans()
    # Set max_results_per_family higher than the number of plans we have
    result = reduce_by_family(plans, max_results_per_family=10)

    # Should return all 5 plans since we have 3 from family_a and 2 from family_b
    assert len(result) == 5
