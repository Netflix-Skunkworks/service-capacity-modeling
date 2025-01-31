try:
    from importlib import resources
    from importlib import import_module
except ImportError:
    import importlib_resources as resources  # type: ignore[no-redef]

    import_module = __import__  # type: ignore[assignment]

from typing import Dict, List
from service_capacity_modeling.hardware import load_hardware_from_disk
from pathlib import Path

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
    for fd in shapes.glob("**/*.json"):
        shape = fd.stem

        print(f"Loading shape={shape} from {Path(shapes, shape)}")
        groups = group_profile_paths(Path(shapes.parent, "pricing", shape))
        for pricing_name, pricing_paths in groups.items():
            print(f"Loading {pricing_name} -> {pricing_paths}")
            ghw = load_hardware_from_disk(price_paths=pricing_paths, shape_path=fd)
            common_profiles[f"{shape}-{pricing_name}"] = ghw
