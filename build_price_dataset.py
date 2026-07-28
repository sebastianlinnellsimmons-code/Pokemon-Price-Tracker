"""
Pokemon Card Price Dataset Builder
------------------------------------
Pulls today's price snapshot from pokemontcg.io and appends it to a local
CSV store. Run this once per day (see scheduling options at the bottom of
this file) and the store grows into a real historical dataset over time.

Why this is its own script: pokemontcg.io has no "give me the price on a
past date" endpoint — only "what is the price right now." So there is no
way to backfill history; the only historical data that will ever exist is
whatever this script has captured, one day at a time, going forward.

Usage:
    export POKEMONTCG_API_KEY="your-key-here"
    python build_price_dataset.py --store prices_store.csv

Re-running this on the same day is safe — it detects that today's
snapshot already exists and skips instead of duplicating rows.
"""

import argparse
import os
import sys
import time
from datetime import date

import numpy as np
import pandas as pd
import requests

API_BASE = "https://api.pokemontcg.io/v2/cards"


def fetch_all_cards(api_key: str, page_size: int = 250, max_retries: int = 5):
    """
    Paginate through the full card list, one page at a time, so the whole
    ~18k-card response is never held in memory at once.

    pokemontcg.io is a community-run API and its own docs note that 5xx
    responses ("an error with the Pokemon TCG API servers") happen — this
    is not something a retry-once client can assume away. We back off with
    exponential wait on both 429 (rate limit) and any 5xx server error, and
    if a single page still won't succeed after max_retries, we skip that
    page and keep going rather than losing the entire day's fetch to one
    server hiccup on one page out of ~75.
    """
    page = 1
    headers = {"X-Api-Key": api_key} if api_key else {}
    skipped_pages = []
    while True:
        page_data = None
        for attempt in range(max_retries):
            try:
                resp = requests.get(
                    API_BASE,
                    headers=headers,
                    params={"page": page, "pageSize": page_size},
                    timeout=30,
                )
            except requests.exceptions.RequestException as e:
                wait = 2 ** attempt
                print(f"Network error on page {page} (attempt {attempt+1}): {e}. Retrying in {wait}s.", file=sys.stderr)
                time.sleep(wait)
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                wait = 2 ** attempt
                print(f"HTTP {resp.status_code} on page {page} (attempt {attempt+1}). Retrying in {wait}s.", file=sys.stderr)
                time.sleep(wait)
                continue

            resp.raise_for_status()  # any other 4xx is a real problem (bad key, bad params) — surface it
            page_data = resp.json().get("data", [])
            break
        else:
            print(f"Page {page} failed after {max_retries} attempts (server-side errors). Skipping this page for today.", file=sys.stderr)
            skipped_pages.append(page)
            page += 1
            # give the API a moment before hammering the next page too
            time.sleep(2)
            # heuristic stop: if we've never gotten a successful page yet
            # and we're already several pages of pure failure deep, the
            # API/key is probably down rather than flaky — stop instead
            # of looping forever
            if page > 5 and len(skipped_pages) == page - 1:
                raise RuntimeError(
                    f"First {page-1} pages all failed — the API or key is likely "
                    "unavailable rather than momentarily flaky. Aborting this run; "
                    "try again later."
                )
            continue

        if not page_data:
            break  # no more pages — normal end of pagination
        yield from page_data
        page += 1

    if skipped_pages:
        print(f"Note: {len(skipped_pages)} page(s) skipped after repeated server errors: {skipped_pages}", file=sys.stderr)


def flatten_snapshot(cards, snapshot_date: str) -> pd.DataFrame:
    """
    Flatten nested per-card JSON into flat rows: one row per (card, variant).
    Handles two real, observed edge cases in this API:
      - some cards have no "tcgplayer" block at all (e.g. foreign-market-only
        older prints), so .get() with defaults everywhere rather than
        assuming the key exists
      - sub-fields like "directLow" are frequently null; we only require
        "market"/"trendPrice" to be present since those are what we train on
    """
    rows = []
    for card in cards:
        base = {
            "card_id": card.get("id"),
            "name": card.get("name"),
            "set": (card.get("set") or {}).get("name"),
            "rarity": card.get("rarity"),
            "release_date": (card.get("set") or {}).get("releaseDate"),
        }

        tcg = (card.get("tcgplayer") or {}).get("prices") or {}
        for variant, price_fields in tcg.items():
            if not price_fields:
                continue
            market = price_fields.get("market")
            if market is None:
                continue
            rows.append({
                **base, "source": "tcgplayer", "variant": variant, "price": market,
                "avg1": np.nan, "avg7": np.nan, "avg30": np.nan, "date": snapshot_date,
            })

        cm = (card.get("cardmarket") or {}).get("prices") or {}
        if cm and cm.get("trendPrice") is not None:
            rows.append({
                **base, "source": "cardmarket", "variant": "normal", "price": cm["trendPrice"],
                "avg1": cm.get("avg1"), "avg7": cm.get("avg7"), "avg30": cm.get("avg30"),
                "date": snapshot_date,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # downcast dtypes immediately rather than after the fact — this matters
    # once the store has months of history across thousands of card/variant
    # combinations, since object/float64 columns waste memory fast
    for col in ("card_id", "name", "set", "rarity", "source", "variant"):
        df[col] = df[col].astype("category")
    df["price"] = pd.to_numeric(df["price"], downcast="float")
    return df


def already_fetched_today(store_path: str, snapshot_date: str) -> bool:
    """
    Idempotency check: if today's date already appears in the store, skip.
    Reads only the date column rather than the whole file, since the store
    can grow to millions of rows after months of daily snapshots.
    """
    if not os.path.exists(store_path):
        return False
    try:
        dates = pd.read_csv(store_path, usecols=["date"])["date"]
    except (ValueError, pd.errors.EmptyDataError):
        return False
    return snapshot_date in dates.values


def atomic_append(new_rows: pd.DataFrame, store_path: str):
    """
    Write-then-rename instead of appending in place, so a crash or power
    loss mid-write can never leave the store in a half-written, corrupt
    state. The existing store is only ever touched by a single, final,
    atomic os.replace() call.
    """
    tmp_path = store_path + ".tmp"
    if os.path.exists(store_path):
        existing = pd.read_csv(store_path)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined.to_csv(tmp_path, index=False)
    os.replace(tmp_path, store_path)  # atomic on POSIX and Windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-key", default=os.environ.get("POKEMONTCG_API_KEY"),
        help="Defaults to the POKEMONTCG_API_KEY env var. Avoid passing keys "
             "directly on the command line, since they land in shell history.",
    )
    parser.add_argument("--store", default="prices_store.csv", help="Path to the accumulating CSV store")
    parser.add_argument("--flush-every", type=int, default=500, help="Cards buffered per flatten batch")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit(
            "No API key found. Set POKEMONTCG_API_KEY as an environment "
            "variable, or pass --api-key (less recommended)."
        )

    snapshot_date = date.today().isoformat()

    if already_fetched_today(args.store, snapshot_date):
        print(f"Already have a snapshot for {snapshot_date} in {args.store}. Skipping.")
        return

    cards_buffer, dfs = [], []
    total_cards = 0
    for card in fetch_all_cards(args.api_key):
        cards_buffer.append(card)
        total_cards += 1
        if len(cards_buffer) >= args.flush_every:
            dfs.append(flatten_snapshot(cards_buffer, snapshot_date))
            cards_buffer = []
    if cards_buffer:
        dfs.append(flatten_snapshot(cards_buffer, snapshot_date))

    non_empty = [d for d in dfs if not d.empty]
    if not non_empty:
        print("No price rows fetched — check your API key and network connection.")
        return

    new_rows = pd.concat(non_empty, ignore_index=True)
    atomic_append(new_rows, args.store)
    print(f"Fetched {total_cards} cards, wrote {len(new_rows)} price rows for {snapshot_date} to {args.store}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# SCHEDULING OPTIONS (pick one)
# ---------------------------------------------------------------------------
#
# Option A — cron (Mac/Linux), runs at 6am daily:
#   crontab -e
#   0 6 * * * cd /path/to/project && POKEMONTCG_API_KEY=yourkey python3 build_price_dataset.py --store prices_store.csv >> fetch.log 2>&1
#
# Option B — Windows Task Scheduler:
#   Create a Basic Task -> Trigger: Daily -> Action: start a program
#   Program: python.exe   Arguments: build_price_dataset.py --store prices_store.csv
#   Set the POKEMONTCG_API_KEY environment variable in the task's settings.
#
# Option C — GitHub Actions (runs in the cloud, works even if your computer
# is off; recommended if you want this to just keep running unattended).
# See build_price_dataset_workflow.yml alongside this file.