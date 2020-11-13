import math
from typing import List
from typing import Sequence

from service_capacity_modeling.models import ClusterCapacity


def reduce_by_family(clusters: Sequence[ClusterCapacity]) -> List[ClusterCapacity]:
    """Groups a potential set of clusters by hardware family sorted by cost.

    Useful for showing different family options.
    """
    families = set()
    result: List[ClusterCapacity] = []
    for topo in sorted(clusters, key=lambda x: x.annual_cost):
        if topo.instance.family in families:
            continue
        families.add(topo.instance.family)
        result.append(topo)
    return result


# https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-or-equal-to-n-in-python
def next_power_of_2(y: float) -> int:
    x = int(y)
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def next_n(x: float, n: float) -> int:
    return int(math.ceil(x / n)) * int(n)
