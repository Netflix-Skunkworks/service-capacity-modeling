import math
from typing import List
from typing import Sequence

from service_capacity_modeling.models import Clusters


def reduce_by_family(clusters: Sequence[Clusters]) -> List[Clusters]:
    """Groups a potential set of clusters by hardware family sorted by cost.

    Useful for showing different family options.
    """
    zonal_families = set()
    regional_families = set()

    result: List[Clusters] = []
    for topo in sorted(clusters, key=lambda x: x.total_annual_cost.mid):
        regional_type, zonal_type = tuple(), tuple()
        if topo.regional:
            regional_type = tuple(sorted({c.instance.family for c in topo.regional}))
            zonal_families.add(zonal_type)
        if topo.zonal:
            zonal_type = tuple(sorted({c.instance.family for c in topo.zonal}))
            zonal_families.add(zonal_type)

        if regional_type in regional_families and zonal_type in zonal_families:
            continue

        if regional_type or zonal_type:
            result.append(topo)
    return result


# https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-or-equal-to-n-in-python
def next_power_of_2(y: float) -> int:
    x = int(y)
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def next_n(x: float, n: float) -> int:
    return int(math.ceil(x / n)) * int(n)
