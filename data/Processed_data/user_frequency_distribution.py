"""Summarize and plot the number of smart-meter records per user.

By default, this script reads ``Good_Data.csv`` from the same directory and
saves the histogram as publication-quality ``Fig2.tiff`` and a
GitHub-previewable ``Fig2.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "Good_Data.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "Fig2.tiff"
USER_COLUMN = "user key"


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Count records per user and plot their frequency distribution."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output TIFF figure; a PNG copy is saved beside it "
            f"(default: {DEFAULT_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after saving it.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the processed data, report user frequencies, and create Fig. 2."""
    args = parse_args()

    if args.bins <= 0:
        raise ValueError("--bins must be greater than zero")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    good_data = pd.read_csv(args.input, usecols=[USER_COLUMN])
    user_record_counts = good_data[USER_COLUMN].value_counts()

    if user_record_counts.empty:
        raise ValueError(f"No non-empty values were found in column '{USER_COLUMN}'")

    print("Summary of Data Point Distribution per User:")
    print(user_record_counts.describe())

    top_user = user_record_counts.idxmax()
    top_user_data_points = int(user_record_counts.max())
    print(
        f"\nUser with the most data points: {top_user} "
        f"({top_user_data_points} data points)"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(user_record_counts, bins=args.bins, color="skyblue", edgecolor="black")
    ax.set_xlabel("Number of Data Points")
    ax.set_ylabel("Number of Users")
    ax.grid(axis="y")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    png_output = args.output.with_suffix(".png")
    fig.savefig(args.output, dpi=300, format="tiff", bbox_inches="tight")
    fig.savefig(png_output, dpi=300, format="png", bbox_inches="tight")
    print(f"TIFF figure saved to: {args.output}")
    print(f"PNG figure saved to: {png_output}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
