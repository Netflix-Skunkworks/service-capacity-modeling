#!/usr/bin/python3
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from instance_families import instance_families


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


def main(debug: bool = True, execute: bool = False):
    # Get the path where shape files should be stored
    expected_path = Path(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "hardware",
        "profiles",
        "shapes",
        "aws",
    )

    # Check which families need to be generated
    missing_families = {}
    for family, params in instance_families.items():
        filename = f"auto_{family}.json"
        expected_file = expected_path / filename

        if not expected_file.exists():
            print(f"Missing {family} in {expected_file}")
            missing_families[family] = params

    if not missing_families:
        print("All instance family files are present.")
        return

    # Build commands for missing families
    commands = {}
    for family, params in missing_families.items():
        cmd = build_command(family, params, expected_path)
        commands[family] = cmd

        if debug:
            print(f"\nWould run for {family}:")
            print(" ".join(cmd))

    # Execute commands if requested
    if execute:
        for family, cmd in commands.items():
            print(f"\nGenerating {family}...")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"Successfully generated {family}")
                if debug:
                    print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Error generating {family}: {e}")
                print(e.stderr)
    elif not debug:
        print("\nRun with --execute to generate the missing files")


if __name__ == "__main__":
    # Parse arguments to determine whether to execute commands
    execute_mode = "--execute" in sys.argv
    debug_mode = "--debug" in sys.argv or not execute_mode

    main(debug=debug_mode, execute=execute_mode)
