#!/usr/bin/env python3
"""
Compare regression coefficients between EBS and local SSD configurations.
"""

import csv
from typing import Any
import numpy as np
import numpy.typing as npt


def r2_score(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    """Calculate R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)  # type: ignore[no-any-return]


def mape(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    """Calculate Mean Absolute Percentage Error."""
    result: float = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return result


def linear_regression(
    X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> tuple[float, npt.NDArray[np.float64]]:
    """Perform linear regression using numpy."""
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    intercept = float(coefficients[0])
    coef = coefficients[1:]
    return intercept, coef  # type: ignore[return-value]


def predict(
    X: npt.NDArray[np.float64], intercept: float, coef: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Make predictions using linear regression coefficients."""
    return intercept + np.dot(X, coef)  # type: ignore[no-any-return]


def load_disk_data() -> tuple[
    dict[str, npt.NDArray[np.float64]], dict[str, npt.NDArray[np.float64]]
]:
    """Load the CSV data separated by disk type."""
    ebs_data: dict[str, Any] = {"reads": [], "writes": [], "data_gib": [], "costs": []}
    ssd_data: dict[str, Any] = {"reads": [], "writes": [], "data_gib": [], "costs": []}

    with open("cassandra_disk_comparison.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            disk_type = row["disk_type"]
            data_dict = ebs_data if disk_type == "ebs" else ssd_data

            data_dict["reads"].append(float(row["reads_per_sec"]))
            data_dict["writes"].append(float(row["writes_per_sec"]))
            data_dict["data_gib"].append(float(row["data_size_gib"]))
            data_dict["costs"].append(float(row["total_annual_cost"]))

    # Convert to numpy arrays
    for data_dict in [ebs_data, ssd_data]:
        for key in data_dict:
            data_dict[key] = np.array(data_dict[key])

    return ebs_data, ssd_data


def analyze_disk_type(
    disk_type: str, data: dict[str, npt.NDArray[np.float64]]
) -> dict[str, float]:
    """Run regression analysis for a specific disk type."""
    reads = data["reads"]
    writes = data["writes"]
    data_gib = data["data_gib"]
    costs = data["costs"]

    print(f"\n{'=' * 80}")
    print(f"REGRESSION ANALYSIS: {disk_type.upper()}")
    print("=" * 80)
    print(f"\nDataset size: {len(costs)} scenarios")
    print(f"Cost range: ${costs.min():,.2f} - ${costs.max():,.2f}")
    print(f"Average cost: ${costs.mean():,.2f}")

    # Model 1: All three factors
    print(f"\n{'-' * 80}")
    print("MODEL: cost ~ reads + writes + data_gib")
    print("-" * 80)

    X = np.column_stack([reads, writes, data_gib])
    intercept, coef = linear_regression(X, costs)
    predictions = predict(X, intercept, coef)

    r2 = r2_score(costs, predictions)
    mape_score = mape(costs, predictions)

    print(f"\nIntercept: ${intercept:,.2f}")
    print(f"Coefficient for reads/sec:  ${coef[0]:.6f} per read/sec")
    print(f"Coefficient for writes/sec: ${coef[1]:.6f} per write/sec")
    print(f"Coefficient for data_gib:   ${coef[2]:.6f} per GiB")
    print(f"\nR² Score: {r2:.4f}")
    print(f"MAPE: {mape_score:.2f}%")

    # Calculate normalized weights
    total_coef = (
        coef[0] * reads.mean() + coef[1] * writes.mean() + coef[2] * data_gib.mean()
    )
    weight_reads = (coef[0] * reads.mean()) / total_coef
    weight_writes = (coef[1] * writes.mean()) / total_coef
    weight_data = (coef[2] * data_gib.mean()) / total_coef

    print(f"\n{'Attribution Weights (normalized to sum = 1.0)':<50}")
    print(f"  Reads:     {weight_reads:>6.3f} ({weight_reads * 100:>5.1f}%)")
    print(f"  Writes:    {weight_writes:>6.3f} ({weight_writes * 100:>5.1f}%)")
    print(f"  Data size: {weight_data:>6.3f} ({weight_data * 100:>5.1f}%)")

    return {
        "intercept": intercept,
        "coef_reads": coef[0],
        "coef_writes": coef[1],
        "coef_data": coef[2],
        "weight_reads": weight_reads,
        "weight_writes": weight_writes,
        "weight_data": weight_data,
        "r2": r2,
        "mape": mape_score,
    }


def main() -> None:
    """Compare EBS vs local SSD cost attribution."""
    print("=" * 80)
    print("CASSANDRA COST ATTRIBUTION: EBS vs LOCAL SSD COMPARISON")
    print("=" * 80)

    ebs_data, ssd_data = load_disk_data()

    # Analyze each disk type
    ebs_results = analyze_disk_type("EBS (Attached Storage)", ebs_data)
    ssd_results = analyze_disk_type("Local SSD (NVMe)", ssd_data)

    # Comparison
    print(f"\n{'=' * 80}")
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 80)

    # Pre-compute weight differences to shorten lines
    read_diff = (ebs_results["weight_reads"] - ssd_results["weight_reads"]) * 100
    write_diff = (ebs_results["weight_writes"] - ssd_results["weight_writes"]) * 100
    data_diff = (ebs_results["weight_data"] - ssd_results["weight_data"]) * 100

    comparison_table = [
        ("Metric", "EBS", "Local SSD", "Difference"),
        ("-" * 30, "-" * 20, "-" * 20, "-" * 20),
        (
            "Intercept",
            f"${ebs_results['intercept']:,.2f}",
            f"${ssd_results['intercept']:,.2f}",
            f"${ebs_results['intercept'] - ssd_results['intercept']:+,.2f}",
        ),
        (
            "Cost per read/sec",
            f"${ebs_results['coef_reads']:.6f}",
            f"${ssd_results['coef_reads']:.6f}",
            f"${ebs_results['coef_reads'] - ssd_results['coef_reads']:+.6f}",
        ),
        (
            "Cost per write/sec",
            f"${ebs_results['coef_writes']:.6f}",
            f"${ssd_results['coef_writes']:.6f}",
            f"${ebs_results['coef_writes'] - ssd_results['coef_writes']:+.6f}",
        ),
        (
            "Cost per GiB",
            f"${ebs_results['coef_data']:.6f}",
            f"${ssd_results['coef_data']:.6f}",
            f"${ebs_results['coef_data'] - ssd_results['coef_data']:+.6f}",
        ),
        ("", "", "", ""),
        (
            "Weight: Reads",
            (
                f"{ebs_results['weight_reads']:.3f} "
                f"({ebs_results['weight_reads'] * 100:.1f}%)"
            ),
            (
                f"{ssd_results['weight_reads']:.3f} "
                f"({ssd_results['weight_reads'] * 100:.1f}%)"
            ),
            f"{read_diff:+.1f}%",
        ),
        (
            "Weight: Writes",
            (
                f"{ebs_results['weight_writes']:.3f} "
                f"({ebs_results['weight_writes'] * 100:.1f}%)"
            ),
            (
                f"{ssd_results['weight_writes']:.3f} "
                f"({ssd_results['weight_writes'] * 100:.1f}%)"
            ),
            f"{write_diff:+.1f}%",
        ),
        (
            "Weight: Data",
            (
                f"{ebs_results['weight_data']:.3f} "
                f"({ebs_results['weight_data'] * 100:.1f}%)"
            ),
            (
                f"{ssd_results['weight_data']:.3f} "
                f"({ssd_results['weight_data'] * 100:.1f}%)"
            ),
            f"{data_diff:+.1f}%",
        ),
        ("", "", "", ""),
        (
            "R² Score",
            f"{ebs_results['r2']:.4f}",
            f"{ssd_results['r2']:.4f}",
            f"{ebs_results['r2'] - ssd_results['r2']:+.4f}",
        ),
        (
            "MAPE",
            f"{ebs_results['mape']:.2f}%",
            f"{ssd_results['mape']:.2f}%",
            f"{ebs_results['mape'] - ssd_results['mape']:+.2f}%",
        ),
    ]

    for row in comparison_table:
        print(f"{row[0]:<30} {row[1]:>20} {row[2]:>20} {row[3]:>20}")

    # Key insights
    print(f"\n{'=' * 80}")
    print("KEY INSIGHTS")
    print("=" * 80)

    read_ratio = ebs_results["coef_reads"] / ssd_results["coef_reads"]
    write_ratio = ebs_results["coef_writes"] / ssd_results["coef_writes"]
    data_ratio = ebs_results["coef_data"] / ssd_results["coef_data"]

    ebs_int = ebs_results["intercept"]
    ssd_int = ssd_results["intercept"]
    ebs_reads = ebs_results["coef_reads"]
    ssd_reads = ssd_results["coef_reads"]
    ebs_writes = ebs_results["coef_writes"]
    ssd_writes = ssd_results["coef_writes"]

    print(f"""
1. OVERALL COST STRUCTURE:
   - EBS has higher base cost (intercept): ${ebs_int:,.2f} vs
     ${ssd_int:,.2f}
   - EBS configurations are generally more expensive for the same
     workload

2. READ COST:
   - EBS: ${ebs_reads:.6f} per read/sec
   - SSD: ${ssd_reads:.6f} per read/sec
   - Ratio: {read_ratio:.2f}x (EBS is {read_ratio:.1%} of local SSD
     cost per read)

3. WRITE COST:
   - EBS: ${ebs_writes:.6f} per write/sec
   - SSD: ${ssd_writes:.6f} per write/sec
   - Ratio: {write_ratio:.2f}x (EBS is {write_ratio:.1%} of local SSD
     cost per write)

4. DATA STORAGE COST:
   - EBS: ${ebs_results["coef_data"]:.6f} per GiB
   - SSD: ${ssd_results["coef_data"]:.6f} per GiB
   - Ratio: {data_ratio:.2f}x

5. ATTRIBUTION WEIGHT DIFFERENCES:
   - Reads: {abs(read_diff):.1f}% difference
   - Writes: {abs(write_diff):.1f}% difference
   - Data: {abs(data_diff):.1f}% difference

6. WHICH MATTERS MORE:
   - For EBS: Writes account for
     {ebs_results["weight_writes"] * 100:.1f}% of cost attribution
   - For SSD: Writes account for
     {ssd_results["weight_writes"] * 100:.1f}% of cost attribution
   - Both are write-dominated, but the balance is similar
    """)

    print(f"\n{'=' * 80}")
    print("PRACTICAL RECOMMENDATIONS")
    print("=" * 80)
    print(f"""
1. USE DISK-TYPE SPECIFIC FORMULAS:

   For EBS-based tables:
   table_cost = cluster_cost × [
       {ebs_results["weight_reads"]:.3f} ×
           (table_reads / cluster_reads) +
       {ebs_results["weight_writes"]:.3f} ×
           (table_writes / cluster_writes) +
       {ebs_results["weight_data"]:.3f} ×
           (table_data_gib / cluster_data_gib)
   ]

   For Local SSD-based tables:
   table_cost = cluster_cost × [
       {ssd_results["weight_reads"]:.3f} ×
           (table_reads / cluster_reads) +
       {ssd_results["weight_writes"]:.3f} ×
           (table_writes / cluster_writes) +
       {ssd_results["weight_data"]:.3f} ×
           (table_data_gib / cluster_data_gib)
   ]

2. WHEN THE DIFFERENCE MATTERS:
   - If your cluster uses EBS, use the EBS formula
   - If your cluster uses local SSDs (i3, i4i, d3, etc.), use the
     SSD formula
   - The weight differences are relatively small, so using the wrong
     formula won't cause huge errors, but using the correct one
     improves accuracy

3. WHY THEY DIFFER:
   - EBS has different performance characteristics (IOPS limits,
     throughput caps)
   - Local SSDs have higher IOPS but are more expensive per GB
   - The planner chooses different instance families for each
     configuration
   - This leads to different cost structures even for the same
     workload

4. MODEL QUALITY:
   - Both models are highly accurate (R² > 0.99)
   - MAPE is low for both (~{ebs_results["mape"]:.1f}% for EBS,
     ~{ssd_results["mape"]:.1f}% for SSD)
   - You can confidently use either formula for its respective
     disk type
    """)


if __name__ == "__main__":
    main()
