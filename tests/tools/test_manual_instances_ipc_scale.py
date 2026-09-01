"""Ties manually-curated i3.* shapes' cpu_ipc_scale back to BROADWELL_IPC.

i3 is excluded from auto_shape.py generation (see the commented-out entry in
instance_families.py) and is instead hand-maintained in manual_instances.json.
Unlike the auto_*.json families, manual entries aren't covered by
test_cpu_ipc_scale_invariant.py, so a missing or drifted cpu_ipc_scale here
would silently fall back to Instance's default of 1.0 (Skylake), overstating
i3's real (Broadwell-era) per-vCPU throughput.
"""

import json
from pathlib import Path

from service_capacity_modeling.tools.instance_families import BROADWELL_IPC

SHAPES_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "service_capacity_modeling/hardware/profiles/shapes/aws"
)

I3_FAMILIES = ("i3.xlarge", "i3.2xlarge", "i3.4xlarge")


def test_i3_cpu_ipc_scale_matches_broadwell_ipc() -> None:
    data = json.loads((SHAPES_DIR / "manual_instances.json").read_text())
    instances = data["instances"]

    for name in I3_FAMILIES:
        shape = instances[name]
        assert shape["cpu_ipc_scale"] == BROADWELL_IPC, (
            f"{name}: cpu_ipc_scale={shape.get('cpu_ipc_scale')} does not match "
            f"BROADWELL_IPC={BROADWELL_IPC}"
        )
