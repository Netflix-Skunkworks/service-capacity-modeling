import math
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

from service_capacity_modeling.models import CapacityPlan


def reduce_by_family(
    plans: Iterable[CapacityPlan], max_results_per_family: int = 1
) -> List[CapacityPlan]:
    """Groups a potential set of clusters by hardware family sorted by cost.

    Useful for showing different family options.

    Args:
        plans: Iterable of CapacityPlan objects to filter
        max_results_per_family: Maximum number of results to return per
            family combination
    """
    zonal_families: Dict[Tuple[Tuple[str, str], ...], int] = {}
    regional_families: Dict[Tuple[Tuple[str, str], ...], int] = {}

    result: List[CapacityPlan] = []
    for plan in plans:
        topo = plan.candidate_clusters
        regional_type: Tuple[Tuple[str, str], ...] = tuple()
        zonal_type: Tuple[Tuple[str, str], ...] = tuple()

        if topo.regional:
            regional_type = tuple(
                sorted({(c.cluster_type, c.instance.family) for c in topo.regional})
            )

        if topo.zonal:
            zonal_type = tuple(
                sorted({(c.cluster_type, c.instance.family) for c in topo.zonal})
            )

        # Count how many of each family combination we've seen
        zonal_count = zonal_families.get(zonal_type, 0)
        regional_count = regional_families.get(regional_type, 0)

        # Add the plan if we haven't reached the maximum for either family type
        if (
            zonal_count < max_results_per_family
            or regional_count < max_results_per_family
        ):
            result.append(plan)

            # Update counters
            zonal_families[zonal_type] = zonal_count + 1
            regional_families[regional_type] = regional_count + 1

    return result


# https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-or-equal-to-n-in-python
def next_power_of_2(y: float) -> int:
    x = int(y)
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def next_n(x: float, n: float) -> int:
    return int(math.ceil(x / n)) * int(n)
