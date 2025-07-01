#!/usr/bin/python3
import subprocess
import sys
from pathlib import Path
from typing import Any
from typing import Dict

from instance_families import INSTANCE_TYPES  # pylint: disable=import-error

print("Loaded instance family count =", len(INSTANCE_TYPES))


def get_auto_shape_path() -> Path:
    """Get the path to auto_shape.py script"""
    current_dir = Path(__file__).parent
    return current_dir / "auto_shape.py"


def build_command(family: str, params: Dict[str, Any], output_path: Path) -> list:
    """Build the command to run auto_shape.py with the appropriate parameters"""
    auto_shape_path = get_auto_shape_path()
    cmd = [sys.executable, str(auto_shape_path)]

    # Add optional parameters if they are provided
    if params.get("xl_iops") is not None:
        cmd.extend(["--xl-iops", params["xl_iops"]])

    if params.get("io_latency_curve") is not None:
        cmd.extend(["--io-latency-curve", params["io_latency_curve"]])

    if params.get("cpu_ipc_scale") is not None:
        cmd.extend(["--cpu-ipc-scale", str(params["cpu_ipc_scale"])])

    # Add output path
    cmd.extend(["--output-path", str(output_path)])

    # Add family as positional argument
    cmd.append(family)

    return cmd


def main(debug: bool = True, execute: bool = False, force: bool = False):
    expected_path = (
        Path(__file__).resolve().parent.parent / "hardware/profiles/shapes/aws"
    )

    print("Checking for files in", expected_path)

    # Check which families need to be generated
    missing_families = {
        family: params
        for family, params in INSTANCE_TYPES.items()
        if force or not (expected_path / f"auto_{family}.json").exists()
    }

    if not missing_families:
        print("All instance family files are present.")
        return

    # Build commands for missing families
    commands = {
        family: build_command(family, params, expected_path)
        for family, params in missing_families.items()
    }

    for family, cmd in commands.items():
        if debug:
            print(f"\nWould run for {family}:\n{' '.join(cmd)}")

        if execute:
            print(f"\nGenerating {family}...")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"Successfully generated {family}")
                if debug:
                    print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Error generating {family}: {e}")
                print(e.stderr)

    if not debug and not execute:
        print("\nRun with --execute to generate the missing files")


if __name__ == "__main__":
    # Parse arguments to determine whether to execute commands
    execute_mode = "--execute" in sys.argv
    debug_mode = "--debug" in sys.argv or not execute_mode
    force_mode = "--force" in sys.argv

    main(debug=debug_mode, execute=execute_mode, force=force_mode)
