# Cassandra Cost Attribution: EBS vs Local SSD

## Executive Summary

**Critical Finding:** The disk type significantly affects cost attribution multipliers. You **MUST** use different formulas for EBS vs local SSD clusters.

### Quick Comparison

| Metric | EBS (gp2/gp3) | Local SSD (NVMe) | Winner |
|--------|---------------|------------------|---------|
| **Reads Weight** | 28.9% | 19.9% | - |
| **Writes Weight** | 42.3% | 71.8% | - |
| **Data Weight** | 28.8% | 8.3% | - |
| **Cost per Read** | $0.88/read/sec | $0.39/read/sec | SSD cheaper |
| **Cost per Write** | $1.28/write/sec | $1.41/write/sec | EBS cheaper |
| **Cost per GiB** | $21.26/GiB | $3.96/GiB | SSD cheaper |
| **Model R²** | 0.9612 | 0.9896 | SSD more predictable |

## Cost Attribution Formulas

### For EBS-based Clusters (gp2/gp3)
```
table_cost = cluster_total_cost × [
    0.289 × (table_reads / cluster_reads) +
    0.423 × (table_writes / cluster_writes) +
    0.288 × (table_data_gib / cluster_data_gib)
]
```

**Characteristics:**
- **More balanced** attribution across all three factors
- Data size is a **major cost driver** (28.8%)
- Reads are more expensive (2.25x vs SSD)
- Best for: Workloads with large data and moderate operations

### For Local SSD-based Clusters (i3, i4i, d3, etc.)
```
table_cost = cluster_total_cost × [
    0.199 × (table_reads / cluster_reads) +
    0.718 × (table_writes / cluster_writes) +
    0.083 × (table_data_gib / cluster_data_gib)
]
```

**Characteristics:**
- **Write-dominated** cost model (71.8%)
- Data size has minimal impact (8.3%)
- More predictable costs (higher R²)
- Best for: High-throughput write workloads

## Key Differences Explained

### 1. Why Data Size Matters More for EBS (28.8% vs 8.3%)

**EBS:**
- IOPS are provisioned based on volume size
- Larger data = more EBS volumes = higher IOPS cost
- gp3 pricing: $0.08/GB-month + $0.005 per provisioned IOPS
- The planner provisions more IOPS as data grows

**Local SSD:**
- Fixed IOPS regardless of data size
- Already included in instance cost
- No per-GB IOPS scaling

### 2. Why Writes Dominate Local SSD (71.8% vs 42.3%)

**Local SSD advantages for writes:**
- Much higher IOPS (50,000-100,000 vs 3,000-16,000 for gp3)
- No throughput throttling
- Lower latency enables the planner to use fewer instances
- Write-heavy workloads get bigger efficiency gains

**EBS characteristics:**
- IOPS limited per volume
- Must provision more capacity for high write workloads
- More balanced between compute and storage costs

### 3. Why Reads Cost More on EBS (2.25x)

**EBS read penalty:**
- Higher read latency → more memory needed for caching
- IOPS limits force larger instance sizes
- Must scale up instance type to get more IOPS

**Local SSD read advantage:**
- Much lower latency (NVMe)
- Higher IOPS allow smaller instances
- Less memory needed for read caching

## Practical Examples

### Example 1: Write-Heavy Table
- 50K writes/sec, 5K reads/sec, 500 GiB data
- Cluster: 200K writes/sec, 100K reads/sec, 5000 GiB

**EBS Attribution:**
```
cost = cluster_cost × [
    0.289 × (5K/100K) +
    0.423 × (50K/200K) +
    0.288 × (500/5000)
]
= cluster_cost × [0.0145 + 0.1058 + 0.0288]
= cluster_cost × 0.1491 = 14.9%
```

**Local SSD Attribution:**
```
cost = cluster_cost × [
    0.199 × (5K/100K) +
    0.718 × (50K/200K) +
    0.083 × (500/5000)
]
= cluster_cost × [0.0100 + 0.1795 + 0.0083]
= cluster_cost × 0.1978 = 19.8%
```

**Impact:** Same table gets 33% more cost on SSD cluster (19.8% vs 14.9%)

### Example 2: Large Data, Low Operations
- 1K reads/sec, 1K writes/sec, 2000 GiB data
- Cluster: 50K reads/sec, 50K writes/sec, 5000 GiB

**EBS Attribution:**
```
cost = cluster_cost × [
    0.289 × (1K/50K) +
    0.423 × (1K/50K) +
    0.288 × (2000/5000)
]
= cluster_cost × [0.0058 + 0.0085 + 0.1152]
= cluster_cost × 0.1295 = 13.0%
```

**Local SSD Attribution:**
```
cost = cluster_cost × [
    0.199 × (1K/50K) +
    0.718 × (1K/50K) +
    0.083 × (2000/5000)
]
= cluster_cost × [0.0040 + 0.0144 + 0.0332]
= cluster_cost × 0.0516 = 5.2%
```

**Impact:** Same table gets 60% LESS cost on SSD cluster (5.2% vs 13.0%)

## Decision Tree: Which Formula to Use?

```
Is your cluster using EBS volumes (gp2/gp3)?
├─ YES → Use EBS formula (balanced: 29% / 42% / 29%)
│         Instance types: r6a, m6a, c5a + gp3 volumes
│
└─ NO → Is it using local NVMe SSDs?
          ├─ YES → Use Local SSD formula (write-heavy: 20% / 72% / 8%)
          │         Instance types: i3, i3en, i4i, d3, r5d, c5d, m5d
          │
          └─ UNSURE → Check instance type:
                      - If it has 'd' in the name → Local SSD
                      - If it's i3/i4i series → Local SSD
                      - Otherwise → Probably EBS
```

## When the Difference Is Critical

### Use Disk-Specific Formulas When:

1. **Write-heavy workloads** (>50K writes/sec)
   - Weight difference: 29.5 percentage points
   - Can cause 30%+ attribution error

2. **Large data tables** (>1 TiB)
   - Weight difference: 20.5 percentage points
   - Can cause 20%+ attribution error

3. **Read-heavy workloads** on EBS
   - EBS reads cost 2.25x more than SSD
   - Significant cost differences

### Generic Formula Acceptable When:

1. Quick estimates (ballpark numbers)
2. Balanced workloads (similar reads/writes, moderate data)
3. You don't know the disk type

## Implementation Checklist

- [ ] Identify disk type for each Cassandra cluster
  - Check instance type (i3/i4i = SSD, r6a/m6a = EBS)
  - Or check AWS console for attached volumes

- [ ] Segment clusters by disk type
  - EBS clusters: use EBS formula
  - SSD clusters: use SSD formula

- [ ] Collect per-table metrics
  - Reads/sec (from metrics)
  - Writes/sec (from metrics)
  - Data size (from nodetool tablestats)

- [ ] Apply appropriate formula

- [ ] Validate: sum of table costs = cluster cost

## Data Files

- `cassandra_disk_comparison.csv` - 128 scenarios (64 EBS + 64 SSD)
- `disk_type_regression.py` - Regression analysis script
- `cassandra_disk_comparison.py` - Data generation script

## Regenerating Analysis

```bash
# Generate fresh data
python cassandra_disk_comparison.py > cassandra_disk_comparison.csv

# Run regression analysis
python disk_type_regression.py
```

## Cost Insights

### Why EBS Clusters Are More Expensive

For equivalent workloads, EBS clusters typically cost **1.4-1.5x more** because:
1. Lower IOPS requires more instances
2. Per-GB IOPS provisioning adds cost
3. Higher read latency requires more memory
4. Must overprovision to meet latency SLOs

### Why Local SSD Is More Efficient

1. **Higher IOPS** (50K-100K vs 3K-16K)
2. **Lower latency** (sub-ms vs 1-3ms)
3. **No IOPS provisioning cost**
4. **Better performance-to-cost ratio** for high-throughput workloads

### When to Choose Each

**Choose EBS when:**
- Data size flexibility needed (can grow volumes)
- Lower initial cost preferred
- Moderate IOPS requirements (<10K)
- Need to separate compute from storage

**Choose Local SSD when:**
- High IOPS required (>20K)
- Write-heavy workloads
- Latency-sensitive applications
- Cost-efficiency at scale
