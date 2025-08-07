import re
from typing import Dict
from typing import List
from typing import Tuple

from service_capacity_modeling.hardware import Instance
from service_capacity_modeling.hardware import shapes


def test_r6id() -> None:
    r6id_24xl = shapes.region("us-east-1").instances["r6id.24xlarge"]
    assert r6id_24xl is not None
    assert r6id_24xl.cpu == 96


def get_instance_families() -> Dict[str, List[Tuple[str, str]]]:
    """Extract all instance families from available instances."""
    region = shapes.region("us-east-1")
    families: Dict[str, List[Tuple[str, str]]] = {}
    for instance in region.instances.values():
        families.setdefault(instance.family, []).append((instance.name, instance.size))
    return families


def get_generation_groups() -> Dict[str, List[Tuple[str, int]]]:
    """Group instance families by their base type and sort by generation."""
    families = get_instance_families()
    generation_groups: Dict[str, List[Tuple[str, int]]] = {}

    for family in families:
        # Extract the base type (c, m, r) and generation number
        match = re.match(r"([a-z]+)(\d+)([a-z]*)", family)
        if match:
            base_type, gen_num, variant = match.groups()
            key = f"{base_type}-{variant}" if variant else base_type
            if key not in generation_groups:
                generation_groups[key] = []
            generation_groups[key].append((family, int(gen_num)))

    # Sort each group by generation number
    for group in generation_groups.values():
        group.sort(key=lambda x: x[1])

    return generation_groups


def test_consistent_core_counts_across_generations() -> None:
    """Test that same-sized instances have consistent CPU counts across generations."""
    region = shapes.region("us-east-1")

    # Group instances by size
    instances_by_size: Dict[str, List[Tuple[str, Instance]]] = {}
    for instance in region.instances.values():
        instances_by_size.setdefault(instance.size, []).append(
            (instance.family, instance)
        )

    # Check that core counts are consistent within each size
    for size, instances in instances_by_size.items():
        if len(instances) <= 1:
            continue

        _, base_instance = instances[0]
        base_cpu = base_instance.cpu

        for _, instance in instances[1:]:
            assert instance.cpu == base_cpu, (
                f"CPU count mismatch for size {size}: {base_instance} has {base_cpu} "
                f"cores, but {instance.name} has {instance.cpu} cores"
            )


def test_performance_increases_with_generation() -> None:
    """Test that CPU performance increases with each generation."""
    generation_groups = get_generation_groups()
    region = shapes.region("us-east-1")
    instance_families = get_instance_families()

    failed_msgs: List[str] = []
    for _, generations in generation_groups.items():
        if len(generations) <= 1:
            continue

        # For each generation pair (gen_n, gen_n+1)
        for curr_gen, next_gen in zip(generations, generations[1:]):
            curr_family, next_family = curr_gen[0], next_gen[0]

            # Get a representative instance from each family
            curr_instances = [name for name, _ in instance_families[curr_family]]
            next_instances = [name for name, _ in instance_families[next_family]]

            # Find matching size instances or fallback to the first instance
            matching_sizes = [
                (curr, next)
                for curr in curr_instances
                for next in next_instances
                if curr.split(".")[1] == next.split(".")[1]
            ]

            curr_inst, next_inst = (
                matching_sizes[0]
                if matching_sizes
                else (curr_instances[0], next_instances[0])
            )

            # Calculate performance
            curr_perf = (
                region.instances[curr_inst].cpu_ghz
                * region.instances[curr_inst].cpu_ipc_scale
            )
            next_perf = (
                region.instances[next_inst].cpu_ghz
                * region.instances[next_inst].cpu_ipc_scale
            )

            if not next_perf > curr_perf:
                failed_msgs.append(
                    f"Performance did not increase from {curr_family} ({curr_gen})"
                    + f" to {next_family} ({next_gen}): {curr_inst} perf={curr_perf}"
                    + f" vs {next_inst} perf={next_perf}"
                )

    assert len(failed_msgs) == 0, (
        f"Not all generations passed the performance test, {failed_msgs}."
    )


def test_memory_proportional_to_cpu() -> None:
    """Test that memory is proportional to CPU for instances within the same family."""
    families = get_instance_families()
    region = shapes.region("us-east-1")

    for family, instances in families.items():
        if len(instances) <= 1:
            continue

        # Calculate memory per CPU core for all instances in this family
        mem_to_cpu_ratios: List[Tuple[str, float]] = []
        for instance_name, _ in instances:
            instance = region.instances[instance_name]
            ratio = instance.ram_gib / instance.cpu
            mem_to_cpu_ratios.append((instance_name, ratio))

        # All ratios should be approximately the same within a family
        base_name, base_ratio = mem_to_cpu_ratios[0]
        for name, ratio in mem_to_cpu_ratios[1:]:
            # Calculate relative difference as a percentage
            relative_diff = abs(ratio - base_ratio) / base_ratio * 100
            max_relative_diff = 10  # Allow for a 10% difference
            assert relative_diff < max_relative_diff, (
                f"Memory to CPU ratio mismatch in family {family}: "
                f"{base_name} has {base_ratio:.2f} GB/core, but {name} has "
                f"{ratio:.2f} GB/core (difference: {relative_diff:.2f}%)"
            )


def test_network_bandwidth_scales_with_size() -> None:
    """Test that network bandwidth scales appropriately with instance size."""
    families = get_instance_families()
    region = shapes.region("us-east-1")

    for family, instances in families.items():
        if len(instances) <= 1:
            continue

        # Sort instances by CPU count to establish size order
        instances_by_cpu: List[Tuple[str, int, float]] = []
        for instance_name, _ in instances:
            instance = region.instances[instance_name]
            instances_by_cpu.append((instance_name, instance.cpu, instance.net_mbps))

        instances_by_cpu.sort(key=lambda x: x[1])  # Sort by CPU count

        # Check that network bandwidth increases (or stays the same) with instance size
        for i in range(len(instances_by_cpu) - 1):
            curr_name, curr_cpu, curr_net = instances_by_cpu[i]
            next_name, next_cpu, next_net = instances_by_cpu[i + 1]

            # Allow for some instances to have the same bandwidth
            assert next_net >= curr_net, (
                f"Network bandwidth does not scale properly in family {family}: "
                f"{curr_name} with {curr_cpu} CPUs has {curr_net} Gbit/s, but "
                f"{next_name} with {next_cpu} CPUs has {next_net} Gbit/s"
            )
