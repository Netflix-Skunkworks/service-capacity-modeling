# Read-Only KV Capacity Model: Algorithm & Analysis

## Overview

The Read-Only KV capacity model implements a **partition-aware algorithm** for capacity planning. It's designed for read-only data serving layers (like OODM) that load data from offline sources and serve read traffic with low latency using RocksDB.

## Algorithm Description

### Inputs

| Parameter | Description | Example |
|-----------|-------------|---------|
| `total_num_partitions` | Number of data partitions (required) | 8 |
| `min_replica_count` | Minimum replicas, actual may be higher | 2 |
| `estimated_state_size_gib` | Total data size | 500 GiB |
| `estimated_read_per_second` | Read QPS | 20,000 |
| `max_data_per_node_gib` | Max data per node (default: 2048) | 2048 GiB |
| `rocksdb_block_cache_percent` | Block cache ratio (default: 0.3) | 0.3 |
| `reserved_memory_gib` | Reserved for OS/JVM (default: 8) | 8 GiB |

### Algorithm Steps

```
1. DISK FIRST: Calculate partition placement
   partition_size = data_size / total_num_partitions
   effective_disk = min(instance_disk, max_data_per_node)
   partitions_per_node = floor(effective_disk / (partition_size * disk_buffer))

   If partitions_per_node < 1 → reject instance (can't fit one partition)

2. Calculate nodes for one copy
   nodes_for_one_copy = ceil(total_num_partitions / partitions_per_node)

3. Start with min_replica_count
   replica_count = min_replica_count

4. ITERATE until constraints satisfied:
   count = nodes_for_one_copy * replica_count
   count = max(2, count)  # Minimum 2 for redundancy

   If count > max_regional_size → reject instance

   Check CPU:    count * instance_cpu >= needed_cores
   Check Memory: count * (instance_ram - reserved) >= memory_per_replica * replica_count

   If both satisfied → DONE
   Else → replica_count += 1, repeat
```

### Outputs

| Parameter | Description |
|-----------|-------------|
| `replica_count` | Actual replica count (may exceed min) |
| `partitions_per_node` | Partitions placed on each node |

## Key Design Decisions

1. **Local disks only**: EBS not supported because the algorithm relies on fixed disk capacity to leverage spare space for additional replicas.

2. **Disk-first approach**: Partition placement determines base node count, then CPU/memory constraints may increase replica count.

3. **Minimum 64GB RAM**: Instances with less RAM are filtered out to ensure adequate memory for RocksDB block cache.

4. **Spare disk utilization**: For compute-heavy workloads, the algorithm adds replicas (using spare disk space) to get more CPU/memory.

## Example Analysis

### Workload: Medium Dataset with Variable RPS

**Input parameters:**
- Data size: 500 GiB
- Partitions: 8
- RPS: 8k-30k (mid: 20k)
- Latency target: 2ms
- Read size: 1 KB

**Calculated requirements:**
- Partition size: 500 / 8 = 62.5 GiB
- With disk buffer (1.15x): ~72 GiB per partition
- Memory per replica: 180 GiB (500 GiB × 0.3 block_cache × 1.2 buffer)
- CPU needed: ~39 cores (with 1.5x buffer)

### Results Comparison

#### With min_replica_count=2

| Rank | Instance | Count | CPU | RAM | Disk | Cost/yr |
|------|----------|-------|-----|-----|------|---------|
| #1 | r5d.4xlarge | 4 | 64 | 512 GiB | 2,236 GiB | $15,177 |
| #2 | r6id.8xlarge | 2 | 64 | 488 GiB | 3,540 GiB | $15,942 |

**Why r5d.4xlarge x 4?**
- partitions_per_node = floor(559 / 72) = 7
- nodes_for_one_copy = ceil(8 / 7) = 2
- With RF=2: 2 × 2 = 4 nodes
- CPU: 4 × 16 = 64 ≥ 39 ✓
- Memory: 4 × 120 = 480 ≥ 360 ✓

#### With min_replica_count=3

| Rank | Instance | Count | CPU | RAM | Disk | Cost/yr |
|------|----------|-------|-----|-----|------|---------|
| #1 | r5d.4xlarge | 6 | 96 | 768 GiB | 3,354 GiB | $22,766 |
| #2 | r6id.8xlarge | 3 | 96 | 732 GiB | 5,310 GiB | $23,913 |

### Actual vs Recommended

**Actual cluster:** 3 × i3en.6xlarge
- CPU: 72 cores
- RAM: 549 GiB
- Disk: 41,910 GiB
- Cost: $26,992/yr

**Why i3en.6xlarge x 2 was rejected (RF=2):**
- Memory needed: 360 GiB
- Available: 2 × (183 - 8) = 350 GiB
- 350 < 360 → **Memory constraint failed**

**Recommended (RF=3):** 6 × r5d.4xlarge
- CPU: 96 cores (+33%)
- RAM: 768 GiB (+40%)
- Disk: 3,354 GiB
- Cost: $22,766/yr
- **Savings: $4,226/yr (16%)**

## Testing Plan

### Unit Tests

1. **Partition-aware algorithm tests**
   - Compute-heavy workload increases replica_count
   - Storage-heavy workload uses min_replica_count
   - Partition size affects partitions_per_node
   - Cluster params contain required outputs

2. **Constraint tests**
   - Memory constraint properly enforced
   - CPU constraint properly enforced
   - max_regional_size limit respected

3. **Instance filtering**
   - Local disks required (EBS rejected)
   - Minimum 64GB RAM enforced

### Property-Based Tests

Universal property tests verify:
- Determinism: same input → same output
- Feasibility: valid input → at least one plan
- QPS monotonicity: higher QPS → more CPU
- Tier capacity: tier 0 ≥ tier 2 in some dimension
- Cost positivity: all plans have positive cost
- Instance count positivity: all clusters have ≥ 1 instance

### Integration Tests

1. Small dataset, high RPS (compute-bound)
2. Large dataset, moderate RPS (storage-bound)
3. Throughput workload (network-bound)
4. Variable RPS with Interval (realistic traffic pattern)

## Conclusion

The partition-aware algorithm effectively balances disk, CPU, and memory constraints while minimizing cost. For the analyzed workload, it identifies potential savings of **16% ($4,226/yr)** compared to the actual deployment by recommending more cost-effective instance types.

Key findings:
1. i3en instances have high disk:RAM ratio, making them suboptimal for memory-intensive workloads
2. r5d instances provide better RAM per dollar for this workload profile
3. The algorithm correctly increases node count to satisfy memory constraints
