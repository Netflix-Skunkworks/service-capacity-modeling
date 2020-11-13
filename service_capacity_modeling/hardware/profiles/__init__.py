try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

from service_capacity_modeling.hardware import load_hardware_from_disk
from service_capacity_modeling.hardware import profiles
from pathlib import Path

common_profiles = {}

with pkg_resources.path(profiles, "shapes") as shapes:
    for fd in shapes.glob("**/*.json"):
        shape = fd.stem

        print(f"Loading {shape} from {Path(shapes, shape)}")
        for pricing in Path(shapes.parent, "pricing", shape).glob("**/*.json"):
            print(f"Loading {pricing}")
            ghw = load_hardware_from_disk(price_path=pricing, shape_path=fd)
            common_profiles[f"{shape}-{pricing.stem}"] = ghw
