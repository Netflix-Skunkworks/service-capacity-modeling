# Contains definitions for AWS instance families.
# These are _relative_ IPC values with Skylake as a baseline.
# They are not targeted to a specific workload, but instead an average
# typical value across a broad range of tests such as those used
# in common online benchmarks.
#
from typing import Any
from typing import Dict

# Intel
HASWELL_IPC = 0.85
SKYLAKE_IPC = 1.0
ICELAKE_IPC = SKYLAKE_IPC * 1.15
SAPPHIRE_RAPIDS_IPC = ICELAKE_IPC * 1.12
EMERALD_RAPIDS_IPC = SAPPHIRE_RAPIDS_IPC * 1.03
GRANITE_RAPIDS_IPC = SAPPHIRE_RAPIDS_IPC * 1.10

# AMD
ROME_IPC = 1.03
MILAN_IPC = SKYLAKE_IPC * 1.15
GENOA_IPC = MILAN_IPC * 1.13

INSTANCE_TYPES: Dict[str, Dict[str, Any]] = {
    "c5": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": SKYLAKE_IPC,
        "cpu_turbo_single_ghz": 3.9,
        "cpu_turbo_all_ghz": 3.6,
    },
    "c5a": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": ROME_IPC,
    },  # no spinnaker
    "c5d": {
        "iops_per_gib": None,
        "io_latency_curve": "5th-gen-ssd",
        "cpu_ipc_scale": SKYLAKE_IPC,
    },  # no spinnaker
    "c5n": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": SKYLAKE_IPC,
        "cpu_turbo_single_ghz": 3.5,
        "cpu_turbo_all_ghz": 3.4,
    },  # no spinnaker
    "c6a": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": MILAN_IPC,
    },  # no spinnaker
    "c6i": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": ICELAKE_IPC,
        "cpu_turbo_single_ghz": 3.5,
        "cpu_turbo_all_ghz": 3.5,
    },
    "c6id": {
        "iops_per_gib": None,
        "io_latency_curve": "6th-gen-ssd",
        "cpu_ipc_scale": ICELAKE_IPC,
    },  # no spinnaker
    "c7a": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": GENOA_IPC,
        "cpu_turbo_single_ghz": 3.7,
        "cpu_turbo_all_ghz": 3.7,
    },
    "c7i": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": SAPPHIRE_RAPIDS_IPC,
        "cpu_turbo_single_ghz": 3.8,
        "cpu_turbo_all_ghz": 3.2,
    },
    "c8i": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": GRANITE_RAPIDS_IPC,
        "cpu_turbo_single_ghz": 3.9,
        "cpu_turbo_all_ghz": 3.9,
    },
    # "g4ad": {'iops_per_gib': None, 'io_latency_curve': 'ssd', 'cpu_ipc_scale': None},
    # "g4dn": {'iops_per_gib': None, 'io_latency_curve': 'ssd', 'cpu_ipc_scale': None},
    # "g5": {
    #     'iops_per_gib': None,
    #     'io_latency_curve': '5th-gen-ssd',
    #     'cpu_ipc_scale': None
    # },
    # "g6": {
    #     'iops_per_gib': None,
    #     'io_latency_curve': '6th-gen-ssd',
    #     'cpu_ipc_scale': None
    # },
    # "g6e": {
    #     'iops_per_gib': None,
    #     'io_latency_curve': '6th-gen-ssd',
    #     'cpu_ipc_scale': None
    # },
    # "hpc7g": {
    #     'iops_per_gib': None,
    #     'io_latency_curve': '6th-gen-ssd',
    #     'cpu_ipc_scale': None
    # },
    # "i3": {'iops_per_gib': None, 'io_latency_curve': 'ssd', 'cpu_ipc_scale': None},
    "i7i": {
        "iops_per_gib": "170/93.5",
        "io_latency_curve": "7th-gen-ephemeral",
        "cpu_ipc_scale": EMERALD_RAPIDS_IPC,
        "cpu_turbo_single_ghz": 4.0,
        "cpu_turbo_all_ghz": 3.2,
    },
    "i3en": {
        "iops_per_gib": None,
        "io_latency_curve": "ssd",
        "cpu_ipc_scale": SKYLAKE_IPC,
        "cpu_turbo_single_ghz": 3.1,
        "cpu_turbo_all_ghz": 3.1,
    },
    "i4i": {
        "iops_per_gib": None,
        "io_latency_curve": "ssd",
        "cpu_ipc_scale": ICELAKE_IPC,
        "cpu_turbo_single_ghz": 3.5,
        "cpu_turbo_all_ghz": 3.5,
    },
    "m4": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": HASWELL_IPC,
        "cpu_turbo_single_ghz": 3.0,
        "cpu_turbo_all_ghz": 2.6,
    },  # all-core is a guess
    "m5": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": SKYLAKE_IPC,
        "cpu_turbo_single_ghz": 3.5,
        "cpu_turbo_all_ghz": 3.1,
    },
    # exclude m5d and m5dn because they are in the manual list
    # "m5d": {
    #     'iops_per_gib': None,
    #     'io_latency_curve': '5th-gen-ssd',
    #     'cpu_ipc_scale': SKYLAKE_IPC,
    #     'cpu_turbo_single_ghz': 3.5,
    #     'cpu_turbo_all_ghz': 3.1
    # },
    # "m5dn": {
    #     'iops_per_gib': None,
    #     'io_latency_curve': '5th-gen-ssd',
    #     'cpu_ipc_scale': SKYLAKE_IPC,
    #     'cpu_turbo_single_ghz': 3.5,
    #     'cpu_turbo_all_ghz': 3.1
    # },
    "m5n": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": SKYLAKE_IPC,
        "cpu_turbo_single_ghz": 3.5,
        "cpu_turbo_all_ghz": 3.1,
    },
    "m6a": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": MILAN_IPC,
        "cpu_turbo_single_ghz": 3.6,
        "cpu_turbo_all_ghz": 3.6,
    },
    # "m6gd": {
    #     'iops_per_gib': None,
    #     'io_latency_curve': '6th-gen-ssd',
    #     'cpu_ipc_scale': None
    # },
    "m6i": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": ICELAKE_IPC,
        "cpu_turbo_single_ghz": 3.5,
        "cpu_turbo_all_ghz": 3.5,
    },
    "m6id": {
        "iops_per_gib": None,
        "io_latency_curve": "6th-gen-ssd",
        "cpu_ipc_scale": ICELAKE_IPC,
        "cpu_turbo_single_ghz": 3.5,
        "cpu_turbo_all_ghz": 3.5,
    },
    "m6idn": {
        "iops_per_gib": None,
        "io_latency_curve": "6th-gen-ssd",
        "cpu_ipc_scale": ICELAKE_IPC,
        "cpu_turbo_single_ghz": 3.5,
        "cpu_turbo_all_ghz": 3.5,
    },
    "m7a": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": GENOA_IPC,
        "cpu_turbo_single_ghz": 3.7,
        "cpu_turbo_all_ghz": 3.7,
    },  # is this turbo speed correct?
    "m7i": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": SAPPHIRE_RAPIDS_IPC,
        "cpu_turbo_single_ghz": 3.8,
        "cpu_turbo_all_ghz": 3.2,
    },
    "m8i": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": GRANITE_RAPIDS_IPC,
        "cpu_turbo_single_ghz": 3.9,
        "cpu_turbo_all_ghz": 3.9,
    },
    # "mac2-m2pro": {
    #     'iops_per_gib': None,
    #     'io_latency_curve': 'ssd',
    #     'cpu_ipc_scale': None
    # },
    # "p4d": {
    #     'iops_per_gib': None,
    #     'io_latency_curve': '5th-gen-ssd',
    #     'cpu_ipc_scale': None
    # },
    # "p4de": {
    #     'iops_per_gib': None,
    #     'io_latency_curve': '5th-gen-ssd',
    #     'cpu_ipc_scale': None
    # },
    # "p5": {
    #     'iops_per_gib': None,
    #     'io_latency_curve':'6th-gen-ssd',
    #     'cpu_ipc_scale': None
    # },
    # "p5en": {
    #     'iops_per_gib': None,
    #     'io_latency_curve': '6th-gen-ssd',
    #     'cpu_ipc_scale': None
    # },
    "r4": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": HASWELL_IPC,
        "cpu_turbo_single_ghz": 3.0,
        "cpu_turbo_all_ghz": 2.6,
    },  # all-core is a guess
    "r5": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": SKYLAKE_IPC,
        "cpu_turbo_single_ghz": 3.5,
        "cpu_turbo_all_ghz": 3.1,
    },
    # exclude r5d and r5dn because they are in the manual list
    # "r5d": {
    #     "iops_per_gib": None,
    #     "io_latency_curve": "5th-gen-ssd",
    #     "cpu_ipc_scale": SKYLAKE_IPC,
    #     "cpu_turbo_single_ghz": 3.5,
    #     "cpu_turbo_all_ghz": 3.1,
    # },
    # "r5dn": {
    #     "iops_per_gib": None,
    #     "io_latency_curve": "5th-gen-ssd",
    #     "cpu_ipc_scale": SKYLAKE_IPC,
    #     "cpu_turbo_single_ghz": 3.5,
    #     "cpu_turbo_all_ghz": 3.1,
    # },
    "r5n": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": SKYLAKE_IPC,
        "cpu_turbo_single_ghz": 3.5,
        "cpu_turbo_all_ghz": 3.1,
    },
    "r6a": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": MILAN_IPC,
        "cpu_turbo_single_ghz": 3.6,
        "cpu_turbo_all_ghz": 3.6,
    },
    "r6i": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": ICELAKE_IPC,
        "cpu_turbo_single_ghz": 3.5,
        "cpu_turbo_all_ghz": 3.5,
    },
    "r6id": {
        "iops_per_gib": None,
        "io_latency_curve": "6th-gen-ssd",
        "cpu_ipc_scale": ICELAKE_IPC,
        "cpu_turbo_single_ghz": 3.5,
        "cpu_turbo_all_ghz": 3.5,
    },
    "r7a": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": GENOA_IPC,
        "cpu_turbo_single_ghz": 3.7,
        "cpu_turbo_all_ghz": 3.7,
    },
    "r7i": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": SAPPHIRE_RAPIDS_IPC,
        "cpu_turbo_single_ghz": 3.8,
        "cpu_turbo_all_ghz": 3.2,
    },
    "r8i": {
        "iops_per_gib": None,
        "io_latency_curve": None,
        "cpu_ipc_scale": GRANITE_RAPIDS_IPC,
        "cpu_turbo_single_ghz": 3.9,
        "cpu_turbo_all_ghz": 3.9,
    },
    # "t3": {'iops_per_gib': None, 'io_latency_curve': 'ssd', 'cpu_ipc_scale': None},
    # "z1d": {'iops_per_gib': None, 'io_latency_curve': 'ssd', 'cpu_ipc_scale': None}
}
