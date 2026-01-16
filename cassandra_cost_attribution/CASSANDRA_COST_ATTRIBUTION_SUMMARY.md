# Cassandra Cost Attribution Analysis - Summary

## Overview

This analysis uses the service capacity modeling tool to understand how reads/sec, writes/sec, and data size relate to Cassandra cluster costs. The goal is to determine the best metrics for attributing cluster costs to individual tables.

## Files Generated

1. **cassandra_cost_analysis.csv** - 125 capacity planning scenarios with varying workload parameters
2. **analyze_cassandra_costs.py** - Statistical analysis of cost patterns
3. **regression_analysis.py** - Regression models to determine optimal attribution weights
4. **cassandra_cost_analysis.py** - Script to generate the CSV data

## Key Findings

### 1. Model Performance

| Model | R² Score | MAPE | Interpretation |
|-------|----------|------|----------------|
| reads + writes + data_gib | **0.9991** | **4.84%** | ✅ Best - explains 99.9% of cost variance |
| reads + writes only | 0.9986 | 9.73% | Good, but less accurate |
| total_ops only | 0.7931 | 41.96% | ❌ Poor - misses important nuances |

### 2. Cost Driver Weights

Based on regression analysis across 125 scenarios:

- **Writes: 74.1%** - Writes are the dominant cost driver in Cassandra
- **Reads: 24.1%** - Reads are the secondary cost driver
- **Data Size: 1.7%** - Data size has minimal direct impact on cost

**Why writes dominate:**
- Writes require more CPU (compaction, multiple disk writes)
- Writes consume heap memory for memtables
- High write workloads trigger more frequent compactions
- Write amplification (one write becomes multiple disk IOs)

### 3. Recommended Cost Attribution Formula

For attributing a **cluster's total cost** to individual tables:

```
table_cost = cluster_total_cost × [
    0.241 × (table_reads_per_sec / cluster_total_reads_per_sec) +
    0.741 × (table_writes_per_sec / cluster_total_writes_per_sec) +
    0.017 × (table_data_gib / cluster_total_data_gib)
]
```

**Alternative - using raw coefficients:**
```
table_cost = $1,221 +
             $0.46 × table_reads_per_sec +
             $1.41 × table_writes_per_sec +
             $3.31 × table_data_gib
```

## Practical Example

### Scenario
- **Cluster total cost:** $240,000/year
- **Cluster metrics:**
  - Total reads: 100,000 reads/sec
  - Total writes: 50,000 writes/sec
  - Total data: 1,000 GiB

### Table A (high-traffic table)
- Reads: 20,000 reads/sec (20% of cluster)
- Writes: 5,000 writes/sec (10% of cluster)
- Data: 150 GiB (15% of cluster)

**Cost attribution:**
```
table_cost = $240,000 × [
    0.241 × (20,000 / 100,000) +
    0.741 × (5,000 / 50,000) +
    0.017 × (150 / 1,000)
]
= $240,000 × [0.048 + 0.074 + 0.003]
= $240,000 × 0.125
= $30,000/year
```

**Breakdown:**
- Reads contribute: $11,580 (48% of table cost, 4.8% of cluster cost)
- Writes contribute: $17,792 (59% of table cost, 7.4% of cluster cost)
- Data contributes: $627 (2% of table cost, 0.3% of cluster cost)

## Implementation Steps

1. **Collect per-table metrics from your Cassandra cluster:**
   - Reads/sec per table (from metrics/monitoring)
   - Writes/sec per table (from metrics/monitoring)
   - Data size per table (from nodetool tablestats or disk usage)

2. **Calculate cluster totals:**
   - Sum all table reads = cluster_total_reads
   - Sum all table writes = cluster_total_writes
   - Sum all table data = cluster_total_data_gib

3. **Apply the formula to each table:**
   - Use the weighted formula above
   - Verify: sum of all table costs should equal cluster cost

4. **Validate and refine:**
   - Compare attributed costs to intuition
   - Adjust if specific workload characteristics differ significantly
   - Consider separate formulas for very different cluster types (latency vs throughput)

## Important Caveats

1. **These weights are based on Netflix Cassandra model defaults** - Your cluster may have different characteristics
2. **Assumes standard Cassandra configuration** - Non-standard compaction strategies, caching, etc. may change the dynamics
3. **Does not account for:**
   - Partition hotspots (some tables may cause disproportionate load)
   - Schema complexity (wide rows, many columns, etc.)
   - Query complexity (simple gets vs. range scans)
   - Read/write consistency levels

4. **Data size has low weight because:**
   - It's already factored into the operations metrics
   - Disk is relatively cheap compared to CPU
   - The model assumes reasonable data density per node
   - Very large or very small data sizes relative to operations may need adjustment

## Cost Attribution Patterns Observed

1. **Low data, high ops** (e.g., 10 GiB, 1M ops/sec): ~$956K/year
2. **High data, low ops** (e.g., 5000 GiB, 2K ops/sec): ~$21K/year
3. **Write-heavy workloads cost ~3x more** than equivalent read-heavy workloads
4. **Cost scales super-linearly with operations** - doubling ops more than doubles cost

## Recommendations

### For Simple Attribution
Use the weighted formula with the 24.1% / 74.1% / 1.7% split.

### For More Accuracy
1. Segment clusters by type (latency-sensitive vs throughput)
2. Run this analysis separately for each cluster type
3. Use cluster-specific weights

### For Maximum Accuracy
1. Instrument your actual cluster with detailed per-table metrics
2. Collect actual cluster costs
3. Run regression on your real data
4. Use cluster-specific attribution model

## Viewing the Data

To explore the full dataset:
```bash
# View the CSV
cat cassandra_cost_analysis.csv | column -t -s,

# Run the analysis
python analyze_cassandra_costs.py

# Run the regression
python regression_analysis.py
```

## Regenerating the Data

To regenerate with different parameters:
```bash
# Edit the ranges in cassandra_cost_analysis.py
# Then run:
python cassandra_cost_analysis.py > cassandra_cost_analysis.csv
python analyze_cassandra_costs.py
python regression_analysis.py
```
