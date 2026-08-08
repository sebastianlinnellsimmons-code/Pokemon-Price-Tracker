"""
One-time migration: splits an existing monolithic prices_wide_store.csv
into the new one-file-per-day format under data/daily/.

Run this ONCE, locally, against whatever version of prices_wide_store.csv
you still have (e.g. pull the last successfully-pushed version from your
repo, or use a local copy from your own machine if you have one newer).
After this runs, commit the resulting data/daily/ folder and switch your
workflow over to the updated build_price_dataset.py + workflow yml.

Usage:
    python migrate_to_daily_files.py --input prices_wide_store.csv --output-dir data/daily
"""

import argparse
import os

import pandas as pd


def migrate(input_path: str, output_dir: str):
    df = pd.read_csv(input_path, encoding="utf-8-sig", low_memory=False)
    os.makedirs(output_dir, exist_ok=True)

    dates = sorted(df["date"].unique())
    print(f"Found {len(dates)} distinct day(s) in {input_path}")

    written, skipped = 0, 0
    for d in dates:
        out_path = os.path.join(output_dir, f"{d}.csv")
        if os.path.exists(out_path):
            # don't clobber a day that's already been migrated or already
            # exists from a newer run - safe to re-run this script
            print(f"  {d}.csv already exists, skipping")
            skipped += 1
            continue
        day_df = df[df["date"] == d]
        day_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        written += 1

    print(f"Wrote {written} new daily file(s), skipped {skipped} already-existing file(s)")
    print(f"Total rows migrated: {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the existing monolithic CSV")
    parser.add_argument("--output-dir", default="data/daily", help="Directory to write daily files into")
    args = parser.parse_args()
    migrate(args.input, args.output_dir)
