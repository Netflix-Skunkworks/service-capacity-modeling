#!/usr/bin/env python3
"""
Perform regression analysis to determine optimal weights for Cassandra cost attribution.
"""

import csv
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
    # Add intercept column
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    # Solve using least squares
    coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    intercept = float(coefficients[0])
    coef = coefficients[1:]
    return intercept, coef  # type: ignore[return-value]


def predict(
    X: npt.NDArray[np.float64], intercept: float, coef: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Make predictions using linear regression coefficients."""
    return intercept + np.dot(X, coef)  # type: ignore[no-any-return]


def load_data() -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Load the CSV data."""
    reads = []
    writes = []
    data_gib = []
    costs = []

    with open("cassandra_cost_analysis.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reads.append(float(row["reads_per_sec"]))
            writes.append(float(row["writes_per_sec"]))
            data_gib.append(float(row["data_size_gib"]))
            costs.append(float(row["total_annual_cost"]))

    return np.array(reads), np.array(writes), np.array(data_gib), np.array(costs)


# pylint: disable=too-many-locals,too-many-statements
def regression_analysis() -> None:
    """Perform regression analysis on the cost data."""

    reads, writes, data_gib, costs = load_data()

    print("=" * 80)
    print("REGRESSION ANALYSIS FOR CASSANDRA COST ATTRIBUTION")
    print("=" * 80)
    print()

    # Model 1: All three factors
    print("MODEL 1: cost ~ reads + writes + data_gib")
    print("-" * 80)

    X = np.column_stack([reads, writes, data_gib])
    intercept1, coef1 = linear_regression(X, costs)
    predictions1 = predict(X, intercept1, coef1)

    r2_1 = r2_score(costs, predictions1)
    mape_1 = mape(costs, predictions1)

    print(f"Intercept: ${intercept1:,.2f}")
    print(f"Coefficient for reads/sec:  ${coef1[0]:.6f} per read/sec")
    print(f"Coefficient for writes/sec: ${coef1[1]:.6f} per write/sec")
    print(f"Coefficient for data_gib:   ${coef1[2]:.6f} per GiB")
    print(f"\nR² Score: {r2_1:.4f}")
    print(f"MAPE: {mape_1:.2f}%")

    # Calculate normalized weights that sum to 1
    total_coef = (
        coef1[0] * reads.mean() + coef1[1] * writes.mean() + coef1[2] * data_gib.mean()
    )
    weight_reads = (coef1[0] * reads.mean()) / total_coef
    weight_writes = (coef1[1] * writes.mean()) / total_coef
    weight_data = (coef1[2] * data_gib.mean()) / total_coef

    print("\nNormalized weights (sum to 1.0):")
    print(f"  Reads weight:  {weight_reads:.3f}")
    print(f"  Writes weight: {weight_writes:.3f}")
    print(f"  Data weight:   {weight_data:.3f}")

    # Model 2: Reads + Writes only
    print("\n" + "=" * 80)
    print("MODEL 2: cost ~ reads + writes (ignoring data size)")
    print("-" * 80)

    X2 = np.column_stack([reads, writes])
    intercept2, coef2 = linear_regression(X2, costs)
    predictions2 = predict(X2, intercept2, coef2)

    r2_2 = r2_score(costs, predictions2)
    mape_2 = mape(costs, predictions2)

    print(f"Intercept: ${intercept2:,.2f}")
    print(f"Coefficient for reads/sec:  ${coef2[0]:.6f} per read/sec")
    print(f"Coefficient for writes/sec: ${coef2[1]:.6f} per write/sec")
    print(f"\nR² Score: {r2_2:.4f}")
    print(f"MAPE: {mape_2:.2f}%")

    # Model 3: Total ops only
    print("\n" + "=" * 80)
    print("MODEL 3: cost ~ total_ops (reads + writes)")
    print("-" * 80)

    total_ops = reads + writes
    X3 = total_ops.reshape(-1, 1)
    intercept3, coef3 = linear_regression(X3, costs)
    predictions3 = predict(X3, intercept3, coef3)

    r2_3 = r2_score(costs, predictions3)
    mape_3 = mape(costs, predictions3)

    print(f"Intercept: ${intercept3:,.2f}")
    print(f"Coefficient for total_ops: ${coef3[0]:.6f} per op/sec")
    print(f"\nR² Score: {r2_3:.4f}")
    print(f"MAPE: {mape_3:.2f}%")

    # Model comparison
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("-" * 80)
    print(f"\n{'Model':<40} {'R²':<10} {'MAPE':<10}")
    print("-" * 60)
    print(f"{'reads + writes + data_gib':<40} {r2_1:<10.4f} {mape_1:<10.2f}%")
    print(f"{'reads + writes':<40} {r2_2:<10.4f} {mape_2:<10.2f}%")
    print(f"{'total_ops':<40} {r2_3:<10.4f} {mape_3:<10.2f}%")

    print("\n" + "=" * 80)
    print("RECOMMENDED COST ATTRIBUTION FORMULA")
    print("-" * 80)
    print(f"""
For a table in a Cassandra cluster, attribute cost as:

  table_cost = cluster_total_cost × [
      {weight_reads:.3f} × (table_reads / cluster_total_reads) +
      {weight_writes:.3f} × (table_writes / cluster_total_writes) +
      {weight_data:.3f} × (table_data_gib / cluster_total_data_gib)
  ]

Or using raw coefficients:

  table_cost = ${intercept1:,.2f} +
               ${coef1[0]:.6f} × table_reads_per_sec +
               ${coef1[1]:.6f} × table_writes_per_sec +
               ${coef1[2]:.6f} × table_data_gib

Key insights:
- R² of {r2_1:.4f} means the model explains {r2_1 * 100:.1f}% of cost variance
- MAPE of {mape_1:.2f}% means predictions are off by {mape_1:.2f}% on average
- Reads account for approximately {weight_reads * 100:.1f}% of cost drivers
- Writes account for approximately {weight_writes * 100:.1f}% of cost drivers
- Data size accounts for approximately {weight_data * 100:.1f}% of cost drivers
    """)

    # Example calculation
    print("\n" + "=" * 80)
    print("EXAMPLE CALCULATION")
    print("-" * 80)
    print("""
Suppose you have a cluster with:
- Total cluster cost: $240,000/year
- Total reads: 100,000 reads/sec
- Total writes: 50,000 writes/sec
- Total data: 1,000 GiB

And a table with:
- Table reads: 20,000 reads/sec (20% of total)
- Table writes: 5,000 writes/sec (10% of total)
- Table data: 150 GiB (15% of total)

Table cost attribution:
    """)

    example_cluster_cost = 240000
    example_cluster_reads = 100000
    example_cluster_writes = 50000
    example_cluster_data = 1000

    example_table_reads = 20000
    example_table_writes = 5000
    example_table_data = 150

    table_cost_weighted = example_cluster_cost * (
        weight_reads * (example_table_reads / example_cluster_reads)
        + weight_writes * (example_table_writes / example_cluster_writes)
        + weight_data * (example_table_data / example_cluster_data)
    )

    print(f"  = ${example_cluster_cost:,} × [")
    print(
        f"      {weight_reads:.3f} × "
        f"({example_table_reads:,} / {example_cluster_reads:,}) +"
    )
    print(
        f"      {weight_writes:.3f} × "
        f"({example_table_writes:,} / {example_cluster_writes:,}) +"
    )
    print(f"      {weight_data:.3f} × ({example_table_data} / {example_cluster_data})")
    print("    ]")
    print(f"  = ${table_cost_weighted:,.2f}/year")

    pct_by_reads = weight_reads * (example_table_reads / example_cluster_reads)
    pct_by_writes = weight_writes * (example_table_writes / example_cluster_writes)
    pct_by_data = weight_data * (example_table_data / example_cluster_data)
    total_pct = pct_by_reads + pct_by_writes + pct_by_data

    print("\nBreakdown:")
    print(
        f"  Reads contribution:  {pct_by_reads:.1%} of cluster cost = "
        f"${example_cluster_cost * pct_by_reads:,.2f}"
    )
    print(
        f"  Writes contribution: {pct_by_writes:.1%} of cluster cost = "
        f"${example_cluster_cost * pct_by_writes:,.2f}"
    )
    print(
        f"  Data contribution:   {pct_by_data:.1%} of cluster cost = "
        f"${example_cluster_cost * pct_by_data:,.2f}"
    )
    print(
        f"  Total:               {total_pct:.1%} of cluster cost = "
        f"${example_cluster_cost * total_pct:,.2f}"
    )


if __name__ == "__main__":
    regression_analysis()
