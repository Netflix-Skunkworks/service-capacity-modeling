try:
    from importlib import resources
    from importlib import import_module
except ImportError:
    import importlib_resources as resources  # type: ignore[no-redef]

    import_module = __import__  # type: ignore[assignment]

import logging
from typing import Dict, List
from service_capacity_modeling.hardware import load_hardware_from_disk
from pathlib import Path

logger = logging.getLogger(__name__)
current_module = import_module(__name__)
common_profiles = {}


def group_profile_paths(shapes_path: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for f in sorted(shapes_path.glob("**/*.json")):
        prefix = f.stem.split("_")[0]
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(f)
    return groups


with resources.path(  # pylint: disable=deprecated-method
    current_module, "profiles.txt"
) as shape_file:
    shapes = Path(shape_file.parent, "shapes")
    for profile in shapes.iterdir():
        shape_profile = profile.stem
        logger.info(f"Loading shape={shape_profile} from {Path(shapes, profile)}")
        shape_paths = list(profile.glob("**/*.json"))
        pricing_path = Path(shapes.parent, "pricing", shape_profile)
        groups = group_profile_paths(pricing_path)
        for pricing_name, pricing_paths in groups.items():
            logger.info(f"Loading {pricing_name} from {pricing_path}")
            ghw = load_hardware_from_disk(
                price_paths=pricing_paths, shape_paths=shape_paths
            )
            common_profiles[f"{shape_profile}-{pricing_name}"] = ghw
