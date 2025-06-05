import unittest
from decimal import Decimal
from typing import Dict
from typing import List

from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.interface import ZoneClusterCapacity
from service_capacity_modeling.models.utils import reduce_by_family


class TestUtils(unittest.TestCase):
    def setUp(self):
        # Create mock hardware instances with different families
        self.shape_family_a1 = Instance(
            name="family_a.a1",
            family_separator=".",
            cpu=2,
            cpu_ghz=2.4,
            ram_gib=8,
            net_mbps=1000,
        )
        self.shape_family_a2 = Instance(
            name="family_a.a2",
            family_separator=".",
            cpu=4,
            cpu_ghz=2.4,
            ram_gib=16,
            net_mbps=2000,
        )
        self.shape_family_a3 = Instance(
            name="family_a.a3",
            family_separator=".",
            cpu=8,
            cpu_ghz=2.4,
            ram_gib=32,
            net_mbps=4000,
        )

        self.shape_family_b1 = Instance(
            name="family_b.b1",
            family_separator=".",
            cpu=2,
            cpu_ghz=2.4,
            ram_gib=8,
            net_mbps=1000,
        )
        self.shape_family_b2 = Instance(
            name="family_b.b2",
            family_separator=".",
            cpu=4,
            cpu_ghz=2.4,
            ram_gib=16,
            net_mbps=2000,
        )

        # Create capacity plans with different combinations of families
        self.plans = self._create_test_capacity_plans()

    def _create_test_capacity_plans(self) -> List[CapacityPlan]:
        # Create 5 plans - 3 with family_a and 2 with family_b
        plans = []

        # Family A plans
        for i, shape in enumerate(
            [self.shape_family_a1, self.shape_family_a2, self.shape_family_a3]
        ):
            annual_cost = (i + 1) * 100.0  # Different costs

            cluster = ZoneClusterCapacity(
                cluster_type="test_cluster",
                count=i + 1,
                instance=shape,
                annual_cost=annual_cost,
            )

            annual_costs_a: Dict[str, Decimal] = {
                "test_cluster": Decimal(str(annual_cost))
            }

            plans.append(
                CapacityPlan(
                    requirements=Requirements(),
                    candidate_clusters=Clusters(
                        annual_costs=annual_costs_a, zonal=[cluster], regional=[]
                    ),
                    cost=annual_cost,  # Different costs
                    efficiency=1.0,
                )
            )

        # Family B plans
        for i, shape in enumerate([self.shape_family_b1, self.shape_family_b2]):
            annual_cost = (i + 1) * 200.0  # Different costs

            cluster = ZoneClusterCapacity(
                cluster_type="test_cluster",
                count=i + 1,
                instance=shape,
                annual_cost=annual_cost,
            )

            annual_costs_b: Dict[str, Decimal] = {
                "test_cluster": Decimal(str(annual_cost))
            }

            plans.append(
                CapacityPlan(
                    requirements=Requirements(),
                    candidate_clusters=Clusters(
                        annual_costs=annual_costs_b, zonal=[cluster], regional=[]
                    ),
                    cost=annual_cost,  # Different costs
                    efficiency=1.0,
                )
            )

        return plans

    def test_reduce_by_family_default(self):
        """Test that reduce_by_family with default parameter
        returns one plan per family."""
        result = reduce_by_family(self.plans)

        # Should return only 2 plans - one from family_a and
        # one from family_b
        self.assertEqual(len(result), 2)

        # Verify we have one from each family
        families = set()
        for plan in result:
            for cluster in plan.candidate_clusters.zonal:
                families.add(cluster.instance.family)

        self.assertEqual(families, {"family_a", "family_b"})

    def test_reduce_by_family_multiple(self):
        """Test that reduce_by_family with max_per_family > 1
        returns multiple plans per family."""
        result = reduce_by_family(self.plans, max_per_family=2)

        # Should return 4 plans - two from family_a and two from family_b
        self.assertEqual(len(result), 4)

        # Count plans per family
        family_counts = {"family_a": 0, "family_b": 0}
        for plan in result:
            for cluster in plan.candidate_clusters.zonal:
                family_counts[cluster.instance.family] += 1

        # Verify we have exactly 2 from each family
        self.assertEqual(family_counts["family_a"], 2)
        self.assertEqual(family_counts["family_b"], 2)

    def test_reduce_by_family_unlimited(self):
        """Test that reduce_by_family with max_per_family > available plans
        returns all plans."""
        # Set max_per_family higher than the number of plans we have
        result = reduce_by_family(self.plans, max_per_family=10)

        # Should return all 5 plans since we have 3 from family_a and 2 from family_b
        self.assertEqual(len(result), 5)


if __name__ == "__main__":
    unittest.main()
