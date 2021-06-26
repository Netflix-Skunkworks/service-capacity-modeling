try:
    import importlib.resources as pkg_resources
    from importlib import import_module
except ImportError:
    import importlib_resources as pkg_resources  # type: ignore[no-redef]

    import_module = __import__  # type: ignore[assignment]

from service_capacity_modeling.hardware import load_hardware_from_disk
from pathlib import Path


current_module = import_module(__name__)
common_profiles = {}

with pkg_resources.path(current_module, "profiles.txt") as shape_file:
    shapes = Path(shape_file.parent, "shapes")
    for fd in shapes.glob("**/*.json"):
        shape = fd.stem

        print(f"Loading shape={shape} from {Path(shapes, shape)}")
        for pricing in Path(shapes.parent, "pricing", shape).glob("**/*.json"):
            print(f"Loading {pricing}")
            ghw = load_hardware_from_disk(price_path=pricing, shape_path=fd)
            common_profiles[f"{shape}-{pricing.stem}"] = ghw
