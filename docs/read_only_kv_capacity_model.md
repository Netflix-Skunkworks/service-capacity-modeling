# Read-Only KV Capacity Model

## Overview

The Read-Only KV capacity model provides partition-aware capacity planning for read-only data serving layers (OODM). These systems load data from offline sources (e.g., S3) and serve read traffic online using RocksDB as the storage backend.

---

## Problem Statement

Traditional capacity planning approaches treat CPU, memory, and disk as independent constraints. However, for partitioned data systems:

1. **Disk determines data placement** - Each node holds a fixed number of partitions based on disk capacity
2. **Replication provides compute scaling** - Adding replicas adds nodes, which adds CPU capacity
3. **Spare disk enables flexibility** - Unused disk space can host additional replicas

This creates an opportunity: instead of provisioning separately for each resource, we can leverage the relationship between disk, partitions, and replicas.

---

## Algorithm: Partition-Aware Capacity Planning

### Step 1: Disk-First Sizing

Calculate how many partitions fit on each node, then determine nodes needed for one complete copy:

```
partitions_per_node = floor(effective_disk_per_node / partition_size_with_buffer)
nodes_for_one_copy = ceil(total_partitions / partitions_per_node)
```

### Step 2: CPU Scaling via Replication

Start with minimum replica count and increase until CPU requirements are met:

```
Start with min_replica_count
While (total_nodes × cores_per_node) < required_cores:
    replica_count += 1
    total_nodes = nodes_for_one_copy × replica_count
```

### Key Insight

Increasing replica count adds nodes, which simultaneously:
- Adds CPU capacity (more cores)
- Adds disk capacity (for additional data copies)
- Maintains data locality (each replica is a complete copy)

---

## Design Decisions

**Local disks only**
EBS is provisioned exactly for data needs; no spare space to leverage for the partition-aware algorithm.

**Memory not used for sizing**
Working set calculation doesn't work well for large datasets. Instead, instances are filtered to require minimum 64 GiB RAM, which is sufficient for OODM use cases.

**CPU uses sqrt staffing model**
`cores = utilization + QoS × sqrt(utilization)` models queuing behavior and provides headroom for latency SLOs.

**Buffers**
- CPU: 1.5x (50% headroom for traffic spikes)
- Disk: 1.15x (15% headroom for data growth)

---

## Case Study 1: IHS (Storage-Heavy Workload)

### Workload Profile

| Metric | Value |
|--------|-------|
| Data Size | 48 TB |
| Partitions | 512 |
| Read RPS | 20,000 |
| Latency Target | 2 ms |
| Current Cluster | 64× i3en.6xlarge (RF=4) |

### Analysis

```
partition_size = 48,000 GiB / 512 = 93.75 GiB
partition_size_with_buffer = 93.75 × 1.15 = 107.81 GiB
partitions_per_node = floor(2048 / 107.81) = 18
nodes_for_one_copy = ceil(512 / 18) = 29
nodes_for_cpu = ceil(58.5 cores / 12 cores per node) = 5
```

CPU needs only 5 nodes, but disk needs 29 nodes per copy. This is a **storage-heavy** workload.

### Recommendation

| Metric | Current | Recommended |
|--------|---------|-------------|
| Instance | i3en.6xlarge | i3en.3xlarge |
| Count | 64 | 116 |
| RF | 4 | 4 |
| Annual Cost | $575,829 | $521,846 |
| **Savings** | - | **$53,983 (9%)** |

**Why it works:** The 48TB dataset drives node count, not CPU. Smaller instances (i3en.3xlarge with 12 cores) provide sufficient CPU while reducing cost per node.

---

## Case Study 2: Ads Profile (Balanced Workload)

### Workload Profile

| Metric | Value |
|--------|-------|
| Data Size | 1.4 TB |
| Partitions | 8 |
| Read RPS | 20,000 |
| Latency Target | 2 ms |
| Current Cluster | 3× i3en.6xlarge (RF=3) |

### Analysis

```
partition_size = 1,397 GiB / 8 = 174.6 GiB
partition_size_with_buffer = 174.6 × 1.15 = 200.8 GiB
partitions_per_node = floor(2048 / 200.8) = 10
nodes_for_one_copy = ceil(8 / 10) = 1
nodes_for_cpu = ceil(45 cores / 16 cores per node) = 3
```

Disk needs only 1 node per copy, but CPU needs 3 nodes. This is a **CPU-constrained** workload.

### Recommendation

| Metric | Current | Recommended |
|--------|---------|-------------|
| Instance | i3en.6xlarge | i4i.4xlarge |
| Count | 3 | 3 |
| RF | 3 | 3 |
| Annual Cost | $26,992 | $14,525 |
| **Savings** | - | **$12,467 (46%)** |

**Why it works:** Small dataset (1.4TB) fits easily on any instance. CPU is the constraint. i4i.4xlarge (newer generation) provides better performance/cost ratio.

---

## Summary Comparison

| Metric | IHS | Ads Profile |
|--------|-----|-------------|
| Data Size | 48 TB | 1.4 TB |
| Partitions | 512 | 8 |
| Primary Constraint | Disk | CPU |
| Current Annual Cost | $576K | $27K |
| Recommended Savings | 9% | 46% |

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| total_num_partitions | (required) | Total number of partitions for the dataset |
| min_replica_count | 2 | Minimum replicas; actual may be higher for compute-heavy workloads |
| max_data_per_node_gib | 2048 | Maximum data per node (caps effective disk) |
| rocksdb_block_cache_percent | 0.3 | Block cache configuration (informational) |
| reserved_memory_gib | 8.0 | Memory reserved for OS/JVM |

---

## Output Parameters

The model outputs these parameters in `cluster_params` for provisioning:

| Parameter | Description |
|-----------|-------------|
| read-only-kv.replica_count | Actual replica count used |
| read-only-kv.partitions_per_node | Partitions placed on each node |
| read-only-kv.nodes_for_one_copy | Nodes needed for one copy of the dataset |
| read-only-kv.nodes_for_cpu | Nodes needed if CPU was the only constraint |
| read-only-kv.effective_disk_per_node_gib | Effective disk capacity per node |

---

## Future Work

1. **Memory optimization** - Implement better working set estimation for large datasets based on actual access patterns and cache hit rates

2. **Instance generation awareness** - Prefer newer instance generations (i4i over i3en) when performance/cost ratio is better

3. **Multi-region planning** - Extend to cross-region capacity planning with latency considerations

---

## Appendix: Formula Reference

**CPU Cores (sqrt staffing model)**
```
cores = RPS × latency_sec + QoS × sqrt(RPS × latency_sec)
```

**Partition Sizing**
```
partition_size = data_size / total_partitions
partition_size_with_buffer = partition_size × disk_buffer (1.15)
```

**Node Calculation**
```
effective_disk = min(instance_disk, max_data_per_node)
partitions_per_node = floor(effective_disk / partition_size_with_buffer)
nodes_for_one_copy = ceil(total_partitions / partitions_per_node)
total_nodes = nodes_for_one_copy × replica_count
```
