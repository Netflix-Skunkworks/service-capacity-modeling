import math
from typing import Iterable
from typing import List
from typing import Set
from typing import Tuple

from service_capacity_modeling.models import CapacityPlan


def reduce_by_family(plans: Iterable[CapacityPlan]) -> List[CapacityPlan]:
    """Groups a potential set of clusters by hardware family sorted by cost.

    Useful for showing different family options.
    """
    zonal_families: Set[Tuple[Tuple[str, str], ...]] = set()
    regional_families: Set[Tuple[Tuple[str, str], ...]] = set()

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

        if not (zonal_type in zonal_families and regional_type in regional_families):
            result.append(plan)

        regional_families.add(regional_type)
        zonal_families.add(zonal_type)

    return result


# https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-or-equal-to-n-in-python
def next_power_of_2(y: float) -> int:
    x = int(y)
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def next_n(x: float, n: float) -> int:
    return int(math.ceil(x / n)) * int(n)
