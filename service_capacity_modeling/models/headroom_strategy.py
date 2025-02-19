from abc import ABC
from abc import abstractmethod


class HeadroomStrategy(ABC):
    @abstractmethod
    def calculate_reserved_headroom(self, effective_cpu: float) -> float:
        pass


class QueuingBasedHeadroomStrategy(HeadroomStrategy):
    """
    The headroom should typically indicate the percentage of CPU that should be
    reserved to ensure sensible performance profile.

    This calculates headroom using the Erlang-C staffing formula with P_Q=30:

    See /notebooks/headroom-estimator.ipynb for details
    """

    def calculate_reserved_headroom(self, effective_cpu: float) -> float:
        return 0.712 / (effective_cpu**0.448)
