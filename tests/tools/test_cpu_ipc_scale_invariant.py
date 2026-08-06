"""Ties shipped auto_*.json cpu_ipc_scale values back to INSTANCE_TYPES.

deduce_cpu_ipc_scale() computes cpu_ipc_scale = arch_ipc * ht_factor, where
arch_ipc comes from INSTANCE_TYPES[family]["cpu_ipc_scale"] and ht_factor is
1.5 for no-HT shapes (cpu == cpu_cores) or 1.0 for HT shapes. Nothing
previously verified that a *shipped* JSON file actually reflects this
relationship, so a family generated before a deduce_cpu_ipc_scale fix (or
regenerated with a stale override) could silently drift from the constant
that's supposed to back it.
"""
import json
from pathlib import Path

import pytest

from service_capacity_modeling.tools.instance_families import INSTANCE_TYPES

SHAPES_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "service_capacity_modeling/hardware/profiles/shapes/aws"
)


def _auto_generated_families():
    families = []
    for path in sorted(SHAPES_DIR.glob("auto_*.json")):
        family = path.stem[len("auto_") :]
        if family.startswith("db_"):
            continue  # RDS families are keyed without the "db." prefix
        if family in INSTANCE_TYPES:
            families.append(family)
    return families


@pytest.mark.parametrize("family", _auto_generated_families())
def test_cpu_ipc_scale_matches_instance_types(family: str) -> None:
    # deduce_cpu_ipc_scale() falls back to arch_ipc=1.0 when no IPC constant is set.
    arch_ipc = INSTANCE_TYPES[family]["cpu_ipc_scale"] or 1.0
    data = json.loads((SHAPES_DIR / f"auto_{family}.json").read_text())

    mismatches = []
    for name, shape in data["instances"].items():
        ht_factor = 1.5 if shape["cpu"] == shape["cpu_cores"] else 1.0
        expected = arch_ipc * ht_factor
        actual = shape["cpu_ipc_scale"]
        if abs(actual - expected) / expected > 0.02:
            mismatches.append(
                f"{name}: cpu={shape['cpu']} cpu_cores={shape['cpu_cores']} "
                f"shipped cpu_ipc_scale={actual}, expected ~{expected:.4f} "
                f"(arch_ipc={arch_ipc:.4f} * ht_factor={ht_factor})"
            )

    assert not mismatches, (
        f"{family}: shipped cpu_ipc_scale doesn't match INSTANCE_TYPES "
        f"(needs regeneration via tools/auto_shape.py):\n" + "\n".join(mismatches)
    )
