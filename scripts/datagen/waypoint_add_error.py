#!/usr/bin/env python3
"""
Add stochastic errors to waypoint CSV files.
Supports normal, uniform, and Laplace distributions for noise.
"""

import argparse
import os

import numpy as np


def main():  # pylint: disable=too-many-locals, too-many-branches
    """
    Main function to parse arguments and apply noise to waypoints.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_csv", type=str, help="Path to the input waypoint CSV file."
    )
    parser.add_argument(
        "output_csv", type=str, help="Path to save the modified CSV file."
    )
    parser.add_argument(
        "--mean", type=float, default=0.0, help="Mean of the error distribution."
    )
    parser.add_argument(
        "--std",
        type=float,
        default=0.05,
        help="Standard deviation of the normal distribution (or scale for others).",
    )
    parser.add_argument(
        "--dist",
        type=str,
        default="normal",
        choices=["normal", "uniform", "laplace"],
        help="Type of distribution to sample errors from.",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default="x_m,y_m",
        help="Comma-separated names of columns to add noise to (e.g., x_m,y_m,vx_mps).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    if not os.path.exists(args.input_csv):
        print(f"Error: Input file {args.input_csv} does not exist.")
        return

    # Read headers
    headers = []
    with open(args.input_csv, "r", encoding="utf-8") as f:
        for _ in range(3):
            headers.append(f.readline().strip())

    # Column mapping from the 3rd header line (the one containing column names)
    # # s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
    col_names = [c.strip() for c in headers[2].lstrip("#").split(";")]
    target_cols = [c.strip() for c in args.columns.split(",")]

    target_indices = []
    for tc in target_cols:
        if tc in col_names:
            target_indices.append(col_names.index(tc))
        else:
            print(
                f"Warning: Column '{tc}' not found in CSV. Available columns: {col_names}"
            )

    if not target_indices:
        print("Error: No valid target columns found. Aborting.")
        return

    # Load data
    data = np.loadtxt(args.input_csv, delimiter=";", skiprows=3)

    # Sample noise
    num_rows = data.shape[0]
    num_cols_to_noise = len(target_indices)

    if args.dist == "normal":
        noise = np.random.normal(
            args.mean, args.std, size=(num_rows, num_cols_to_noise)
        )
    elif args.dist == "uniform":
        # For uniform, std is used as the half-width around the mean
        noise = np.random.uniform(
            args.mean - args.std,
            args.mean + args.std,
            size=(num_rows, num_cols_to_noise),
        )
    elif args.dist == "laplace":
        noise = np.random.laplace(
            args.mean, args.std, size=(num_rows, num_cols_to_noise)
        )
    else:
        raise ValueError(f"Unsupported distribution: {args.dist}")

    # Apply noise
    data_noisy = data.copy()
    for i, col_idx in enumerate(target_indices):
        data_noisy[:, col_idx] += noise[:, i]

    # Save to output CSV
    with open(args.output_csv, "w", encoding="utf-8") as f:
        for h in headers:
            f.write(h + "\n")
        np.savetxt(f, data_noisy, delimiter="; ", fmt="%.7f")

    print(f"Successfully saved noisy waypoints to {args.output_csv}")
    print(
        f"Added {args.dist} noise (mean={args.mean}, std={args.std}) to columns: {target_cols}"
    )


if __name__ == "__main__":
    main()
