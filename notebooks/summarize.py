#!/usr/bin/env python3
import argparse
import json

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize measurements to an interval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "data",
        metavar="FILE",
        type=str,
        help="the data file that contains newline separated measurements",
    )
    parser.add_argument("--output", choices=["json", "python"], default="python")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.9,
        help="The confidence interval you want to find",
    )
    parser.add_argument(
        "--include-range",
        action="store_true",
        help="Populate the minimum and maximum values",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    confidence = args.confidence
    real_data = np.loadtxt(args.data)

    low_p = 100 * (0 + (1 - confidence) / 2.0)
    high_p = 100 * (1 - (1 - confidence) / 2.0)
    low, high = np.percentile(real_data, [low_p, high_p])
    low = round(low, 2)
    high = round(high, 2)
    mean = round(np.mean(real_data), 2)
    data_min, data_max = np.min(real_data), np.max(real_data)

    if args.output == "python":
        r = (
            f"FixedInterval(low={low}, mid={mean}, high={high},"
            f" confidence={confidence}"
        )

        if args.include_range:
            print(f"{r}, minimum_value={data_min}, maximum_value={data_max})")
        else:
            print(f"{r})")
    else:
        result = {
            "low": low,
            "mid": mean,
            "high": high,
            "confidence": confidence,
        }
        if args.include_range:
            result["minimum_value"] = data_min
            result["maximum_value"] = data_max
        print(json.dumps(result))


if __name__ == "__main__":
    main()
