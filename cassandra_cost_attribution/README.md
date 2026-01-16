# Cassandra Cost Attribution Analysis

This directory contains scripts and analysis for attributing Cassandra cluster costs to individual tables based on their reads/sec, writes/sec, and data size.

## 📊 Key Finding

**Disk type significantly affects cost attribution!** Use different formulas for EBS vs local SSD clusters:

- **EBS:** Balanced model (29% reads, 42% writes, 29% data)
- **Local SSD:** Write-dominated (20% reads, 72% writes, 8% data)

See [EBS_VS_SSD_COST_ATTRIBUTION.md](EBS_VS_SSD_COST_ATTRIBUTION.md) for the complete analysis.

## 📁 File Organization

### Primary Analysis (Disk Type Comparison)

| File | Purpose | Size |
|------|---------|------|
| **`EBS_VS_SSD_COST_ATTRIBUTION.md`** | **Main documentation** - Read this first! | Guide |
| `cassandra_disk_comparison.py` | Generates cost data for EBS vs SSD configurations | Script |
| `cassandra_disk_comparison.csv` | Results: 128 scenarios (64 EBS + 64 SSD) | 13 KB |
| `disk_type_regression.py` | Regression analysis comparing EBS vs SSD | Script |

### Baseline Analysis (All Configurations)

| File | Purpose | Size |
|------|---------|------|
| `CASSANDRA_COST_ATTRIBUTION_SUMMARY.md` | Original analysis (all disk types mixed) | Guide |
| `cassandra_cost_analysis.py` | Generates cost data across wide parameter ranges | Script |
| `cassandra_cost_analysis.csv` | Results: 125 scenarios with mixed disk types | 12 KB |
| `analyze_cassandra_costs.py` | Statistical analysis of cost patterns | Script |
| `regression_analysis.py` | Regression analysis (all disk types) | Script |

## 🚀 Quick Start

### 1. Understand the Analysis

Read the main findings:
```bash
cat EBS_VS_SSD_COST_ATTRIBUTION.md
```

### 2. View the Data

```bash
# Disk comparison data (recommended)
cat cassandra_disk_comparison.csv | column -t -s, | less -S

# Or the original baseline data
cat cassandra_cost_analysis.csv | column -t -s, | less -S
```

### 3. Run the Analysis

```bash
# Disk type comparison (recommended)
python disk_type_regression.py

# Or the original baseline analysis
python regression_analysis.py
python analyze_cassandra_costs.py
```

## 🔄 Reproducing the Analysis

### Option A: Disk Type Comparison (Recommended)

Compare EBS vs local SSD configurations:

```bash
# 1. Generate fresh data (takes ~2-3 minutes)
python cassandra_disk_comparison.py > cassandra_disk_comparison.csv 2> /dev/null

# 2. Run regression analysis
python disk_type_regression.py
```

**Parameters:**
- Read rates: 1K, 10K, 50K, 100K ops/sec
- Write rates: 1K, 10K, 50K, 100K ops/sec
- Data sizes: 100, 500, 1K, 5K GiB
- Disk types: EBS (gp2/gp3) and Local SSD (NVMe)

### Option B: Baseline Analysis

Generate the original analysis across all configurations:

```bash
# 1. Generate data (takes ~3-5 minutes)
python cassandra_cost_analysis.py > cassandra_cost_analysis.csv 2> /dev/null

# 2. Run analyses
python analyze_cassandra_costs.py
python regression_analysis.py
```

**Parameters:**
- Read rates: 1K, 10K, 50K, 100K, 500K ops/sec
- Write rates: 1K, 10K, 50K, 100K, 500K ops/sec
- Data sizes: 10, 100, 500, 1K, 5K GiB

### Customizing Parameters

Edit the scripts to test different workload ranges:

```python
# In cassandra_disk_comparison.py or cassandra_cost_analysis.py
read_rates = [1_000, 10_000, 50_000, 100_000]      # Modify these
write_rates = [1_000, 10_000, 50_000, 100_000]     # Modify these
data_sizes = [100, 500, 1_000, 5_000]              # Modify these
```

## 📈 What the Analysis Does

1. **Generates capacity plans** using the service capacity planner
2. **Varies workload parameters** (reads, writes, data size)
3. **Calculates costs** for each scenario
4. **Performs regression analysis** to determine:
   - Cost per read/sec
   - Cost per write/sec
   - Cost per GiB
   - Attribution weights (what % each factor contributes)

## 🎯 Cost Attribution Formulas

### For EBS-based Clusters
```
table_cost = cluster_total_cost × [
    0.289 × (table_reads / cluster_reads) +
    0.423 × (table_writes / cluster_writes) +
    0.288 × (table_data_gib / cluster_data_gib)
]
```

### For Local SSD-based Clusters
```
table_cost = cluster_total_cost × [
    0.199 × (table_reads / cluster_reads) +
    0.718 × (table_writes / cluster_writes) +
    0.083 × (table_data_gib / cluster_data_gib)
]
```

## 📊 Understanding the Results

### CSV Columns

- `disk_type`: "ebs" or "local_ssd"
- `reads_per_sec`, `writes_per_sec`, `data_size_gib`: Input parameters
- `total_annual_cost`: Cluster cost per year
- `instance_type`, `instance_count`: Recommended cluster configuration
- `total_cpu_cores`, `total_ram_gib`, `total_disk_gib`: Total resources
- `cost_per_read`, `cost_per_write`, `cost_per_gib`, `cost_per_total_ops`: Attribution metrics

### Regression Output

- **R² Score**: How well the model explains cost variance (higher = better, >0.95 is excellent)
- **MAPE**: Mean Absolute Percentage Error (lower = better, <10% is good)
- **Weights**: Normalized coefficients showing relative importance of each factor

## 🔍 Example Usage

You have a Cassandra cluster:
- Total cost: $240,000/year
- Disk type: Local SSD (i4i instances)
- Total reads: 100K/sec, writes: 50K/sec, data: 1000 GiB

Table "users":
- Reads: 20K/sec, writes: 5K/sec, data: 150 GiB

**Attribution:**
```
users_cost = $240,000 × [
    0.199 × (20K/100K) +
    0.718 × (5K/50K) +
    0.083 × (150/1000)
]
= $240,000 × [0.0398 + 0.0718 + 0.0125]
= $240,000 × 0.1241
= $29,784/year
```

## 🛠️ Requirements

- Python 3.9+
- `service_capacity_modeling` package (installed in this repo)
- `numpy` (for regression analysis)

## 📚 Additional Resources

- See the [main project README](../README.md) for service capacity modeling details
- Netflix Cassandra model: `service_capacity_modeling/models/org/netflix/cassandra.py`
- Tests: `tests/netflix/test_cassandra.py`

## 🤝 Contributing

To extend this analysis:

1. **Test different replication factors:**
   ```python
   extra_model_arguments={'copies_per_region': 2}  # RF=2
   ```

2. **Test different regions:**
   ```python
   region="us-west-2"  # Or any other region
   ```

3. **Test different tiers:**
   ```python
   service_tier=0  # Tier 0 (critical)
   service_tier=2  # Tier 2 (non-critical)
   ```

4. **Add more workload patterns:**
   - Test specific read/write ratios
   - Test burst workloads
   - Test seasonal patterns

## 📝 Citation

If you use this analysis in your cost attribution system, please reference:
- Analysis date: January 2025
- Model: Netflix Cassandra Capacity Model
- Tool: service-capacity-modeling

## ❓ Questions?

Issues or suggestions? File an issue in the parent repository.
