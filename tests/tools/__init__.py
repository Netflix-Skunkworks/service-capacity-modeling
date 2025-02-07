import json

mock_r6id_2xl = json.loads(
    """
{
    "InstanceTypes": [
        {
            "InstanceType": "r6id.2xlarge",
            "CurrentGeneration": true,
            "FreeTierEligible": false,
            "SupportedUsageClasses": [
                "on-demand",
                "spot"
            ],
            "SupportedRootDeviceTypes": [
                "ebs"
            ],
            "SupportedVirtualizationTypes": [
                "hvm"
            ],
            "BareMetal": false,
            "Hypervisor": "nitro",
            "ProcessorInfo": {
                "SupportedArchitectures": [
                    "x86_64"
                ],
                "SustainedClockSpeedInGhz": 3.5,
                "Manufacturer": "Intel"
            },
            "VCpuInfo": {
                "DefaultVCpus": 8,
                "DefaultCores": 4,
                "DefaultThreadsPerCore": 2,
                "ValidCores": [
                    2,
                    4
                ],
                "ValidThreadsPerCore": [
                    1,
                    2
                ]
            },
            "MemoryInfo": {
                "SizeInMiB": 65536
            },
            "InstanceStorageSupported": true,
            "InstanceStorageInfo": {
                "TotalSizeInGB": 474,
                "Disks": [
                    {
                        "SizeInGB": 474,
                        "Count": 1,
                        "Type": "ssd"
                    }
                ],
                "NvmeSupport": "required",
                "EncryptionSupport": "required"
            },
            "EbsInfo": {
                "EbsOptimizedSupport": "default",
                "EncryptionSupport": "supported",
                "EbsOptimizedInfo": {
                    "BaselineBandwidthInMbps": 2500,
                    "BaselineThroughputInMBps": 312.5,
                    "BaselineIops": 12000,
                    "MaximumBandwidthInMbps": 10000,
                    "MaximumThroughputInMBps": 1250.0,
                    "MaximumIops": 40000
                },
                "NvmeSupport": "required"
            },
            "NetworkInfo": {
                "NetworkPerformance": "Up to 12.5 Gigabit",
                "MaximumNetworkInterfaces": 4,
                "MaximumNetworkCards": 1,
                "DefaultNetworkCardIndex": 0,
                "NetworkCards": [
                    {
                        "NetworkCardIndex": 0,
                        "NetworkPerformance": "Up to 12.5 Gigabit",
                        "MaximumNetworkInterfaces": 4,
                        "BaselineBandwidthInGbps": 3.125,
                        "PeakBandwidthInGbps": 12.5
                    }
                ],
                "Ipv4AddressesPerInterface": 15,
                "Ipv6AddressesPerInterface": 15,
                "Ipv6Supported": true,
                "EnaSupport": "required",
                "EfaSupported": false,
                "EncryptionInTransitSupported": true,
                "EnaSrdSupported": false
            },
            "PlacementGroupInfo": {
                "SupportedStrategies": [
                    "cluster",
                    "partition",
                    "spread"
                ]
            },
            "HibernationSupported": false,
            "BurstablePerformanceSupported": false,
            "DedicatedHostsSupported": true,
            "AutoRecoverySupported": false,
            "SupportedBootModes": [
                "legacy-bios",
                "uefi"
            ],
            "NitroEnclavesSupport": "supported",
            "NitroTpmSupport": "supported",
            "NitroTpmInfo": {
                "SupportedVersions": [
                    "2.0"
                ]
            },
            "PhcSupport": "unsupported"
        }
    ]
}

"""
)
mock_m7a_12xl = json.loads(
    """
{
    "InstanceTypes": [
        {
            "InstanceType": "m7a.12xlarge",
            "CurrentGeneration": true,
            "FreeTierEligible": false,
            "SupportedUsageClasses": [
                "on-demand",
                "spot"
            ],
            "SupportedRootDeviceTypes": [
                "ebs"
            ],
            "SupportedVirtualizationTypes": [
                "hvm"
            ],
            "BareMetal": false,
            "Hypervisor": "nitro",
            "ProcessorInfo": {
                "SupportedArchitectures": [
                    "x86_64"
                ],
                "SustainedClockSpeedInGhz": 3.7,
                "Manufacturer": "AMD"
            },
            "VCpuInfo": {
                "DefaultVCpus": 48,
                "DefaultCores": 48,
                "DefaultThreadsPerCore": 1,
                "ValidCores": [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    12,
                    18,
                    24,
                    30,
                    36,
                    42,
                    48
                ],
                "ValidThreadsPerCore": [
                    1
                ]
            },
            "MemoryInfo": {
                "SizeInMiB": 196608
            },
            "InstanceStorageSupported": false,
            "EbsInfo": {
                "EbsOptimizedSupport": "default",
                "EncryptionSupport": "supported",
                "EbsOptimizedInfo": {
                    "BaselineBandwidthInMbps": 15000,
                    "BaselineThroughputInMBps": 1875.0,
                    "BaselineIops": 60000,
                    "MaximumBandwidthInMbps": 15000,
                    "MaximumThroughputInMBps": 1875.0,
                    "MaximumIops": 60000
                },
                "NvmeSupport": "required"
            },
            "NetworkInfo": {
                "NetworkPerformance": "18.75 Gigabit",
                "MaximumNetworkInterfaces": 8,
                "MaximumNetworkCards": 1,
                "DefaultNetworkCardIndex": 0,
                "NetworkCards": [
                    {
                        "NetworkCardIndex": 0,
                        "NetworkPerformance": "18.75 Gigabit",
                        "MaximumNetworkInterfaces": 8,
                        "BaselineBandwidthInGbps": 18.75,
                        "PeakBandwidthInGbps": 18.75
                    }
                ],
                "Ipv4AddressesPerInterface": 30,
                "Ipv6AddressesPerInterface": 30,
                "Ipv6Supported": true,
                "EnaSupport": "required",
                "EfaSupported": false,
                "EncryptionInTransitSupported": true,
                "EnaSrdSupported": true
            },
            "PlacementGroupInfo": {
                "SupportedStrategies": [
                    "cluster",
                    "partition",
                    "spread"
                ]
            },
            "HibernationSupported": false,
            "BurstablePerformanceSupported": false,
            "DedicatedHostsSupported": true,
            "AutoRecoverySupported": true,
            "SupportedBootModes": [
                "legacy-bios",
                "uefi"
            ],
            "NitroEnclavesSupport": "supported",
            "NitroTpmSupport": "supported",
            "NitroTpmInfo": {
                "SupportedVersions": [
                    "2.0"
                ]
            },
            "PhcSupport": "supported"
        }
    ]
}
"""
)

mock_data = {"r6id": mock_r6id_2xl, "m7a": mock_m7a_12xl}
