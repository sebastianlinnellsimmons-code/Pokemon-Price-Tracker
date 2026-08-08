"""
Pokemon Card Price Dataset Builder
------------------------------------
Pulls today's price snapshot from pokemontcg.io and appends it to a local
CSV store, one row per card per day, in the exact column layout below.

IMPORTANT - data source reality check:
pokemontcg.io has no "give me the price on a past date" endpoint - only
"what is the price right now." There is no way to backfill history; the
only historical data that will ever exist is whatever this script has
captured, one day at a time, going forward. Run it daily (see scheduling
notes at the bottom) and the store grows into a real historical dataset.

SCHEMA NOTE:
Real API responses contain tcgplayer variant keys beyond the ones
documented (e.g. "unlimitedHolofoil" alongside "1stEditionHolofoil").
Named variants below get their own column; anything else is preserved
in `other_tcgplayer_variants_json` rather than silently dropped.

Column order (one row per card per day):
  card_id, name, variant, set, rarity, artist, subtype, national_dex_numbers,
  supertype, release_date, price_source, purchase_url,
  tcgplayer_market, tcgplayer_low, tcgplayer_mid, tcgplayer_high,
  tcgplayer_holofoil_market, tcgplayer_reverse_holofoil_market,
  tcgplayer_1st_edition_normal_market, tcgplayer_1st_edition_holofoil_market,
  other_tcgplayer_variants_json,
  cardmarket_average_sell_price_usd, cardmarket_low_price_usd,
  cardmarket_trend_price_usd, cardmarket_reverse_holo_sell_usd,
  cardmarket_avg1_usd, cardmarket_avg7_usd, cardmarket_avg30_usd,
  cardmarket_reverse_holo_avg1_usd, cardmarket_reverse_holo_avg7_usd,
  cardmarket_reverse_holo_avg30_usd,
  eur_usd_rate_used, date

Usage:
    export POKEMONTCG_API_KEY="your-key-here"
    python build_price_dataset.py --store prices_wide_store.csv

Re-running this on the same day is safe - it detects that today's
snapshot already exists and skips instead of duplicating rows.
"""

import argparse
import glob
import json
import os
import sys
import time
from datetime import date

import numpy as np
import pandas as pd
import requests

API_BASE = "https://api.pokemontcg.io/v2/cards"
FX_API = "https://api.frankfurter.app/latest"
FALLBACK_EUR_USD_RATE = 1.08  # rough long-run approximate; only used if the live lookup fails

COLUMN_ORDER = [
    "card_id", "name", "variant", "set", "card_number", "rarity", "artist", "subtype",
    "national_dex_numbers", "supertype", "release_date", "price_source",
    "purchase_url", "image_url", "tcgplayer_updated_at", "cardmarket_updated_at",
    "tcgplayer_market", "tcgplayer_low", "tcgplayer_mid", "tcgplayer_high",
    "tcgplayer_holofoil_market", "tcgplayer_reverse_holofoil_market",
    "tcgplayer_1st_edition_normal_market", "tcgplayer_1st_edition_holofoil_market",
    "other_tcgplayer_variants_json",
    "cardmarket_average_sell_price_usd", "cardmarket_low_price_usd",
    "cardmarket_trend_price_usd", "cardmarket_reverse_holo_sell_usd",
    "cardmarket_avg1_usd", "cardmarket_avg7_usd", "cardmarket_avg30_usd",
    "cardmarket_reverse_holo_avg1_usd", "cardmarket_reverse_holo_avg7_usd",
    "cardmarket_reverse_holo_avg30_usd",
    "eur_usd_rate_used", "date",
]

# tcgplayer variant keys we know about and give an explicit "primary" price to
NAMED_VARIANT_COLUMNS = {
    "holofoil": "tcgplayer_holofoil_market",
    "reverseHolofoil": "tcgplayer_reverse_holofoil_market",
    "1stEditionNormal": "tcgplayer_1st_edition_normal_market",
    "1stEditionHolofoil": "tcgplayer_1st_edition_holofoil_market",
}
# priority order for picking which variant backs the generic Market/Low/Mid/High
# columns - "normal" is what most cards use as their base print
PRIMARY_VARIANT_PRIORITY = ["normal", "unlimited", "holofoil", "1stEditionNormal", "1stEditionHolofoil", "reverseHolofoil"]


# ---------------------------------------------------------------------------
# CURRENCY CONVERSION
# ---------------------------------------------------------------------------
def get_eur_usd_rate(max_retries: int = 3) -> float:
    """
    Fetch today's EUR->USD rate from a free, no-key-required API. Retries
    on transient failure; falls back to a hardcoded approximate rate
    (clearly logged) rather than crashing the whole day's fetch if the
    currency API is down - a stale conversion is better than no data at all,
    and it's logged so it's auditable later.
    """
    for attempt in range(max_retries):
        try:
            resp = requests.get(FX_API, params={"from": "EUR", "to": "USD"}, timeout=15)
            resp.raise_for_status()
            rate = resp.json()["rates"]["USD"]
            if rate and rate > 0:
                return float(rate)
        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            print(f"EUR->USD rate lookup failed (attempt {attempt+1}): {e}", file=sys.stderr)
            time.sleep(1)
    print(f"Falling back to approximate EUR->USD rate: {FALLBACK_EUR_USD_RATE}", file=sys.stderr)
    return FALLBACK_EUR_USD_RATE


def eur_to_usd(value, rate: float):
    """None-safe conversion - never multiply None/NaN, or every cardmarket
    column would raise on the many cards missing that particular field."""
    if value is None:
        return None
    return round(float(value) * rate, 4)


# ---------------------------------------------------------------------------
# FETCH (unchanged pagination/retry logic - the API layer didn't change,
# only what we do with each card once we have it)
# ---------------------------------------------------------------------------
def fetch_all_cards(api_key: str, page_size: int = 250, max_retries: int = 5):
    page = 1
    headers = {"X-Api-Key": api_key} if api_key else {}
    skipped_pages = []
    while True:
        page_data = None
        for attempt in range(max_retries):
            try:
                resp = requests.get(
                    API_BASE, headers=headers,
                    params={"page": page, "pageSize": page_size}, timeout=30,
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

            if resp.status_code == 404:
                # A 404 after we've already succeeded on earlier pages most
                # likely means we've paginated past the real end of the card
                # list (some APIs 404 an out-of-range page instead of
                # returning an empty list) - this is expected, not a failure.
                # But rule out a one-off transient blip first with a single
                # short retry before treating it as the definitive end,
                # since silently stopping pagination early on a genuine
                # transient error would quietly truncate the day's dataset.
                if attempt == 0:
                    print(f"HTTP 404 on page {page} - retrying once to rule out a transient blip.", file=sys.stderr)
                    time.sleep(1)
                    continue
                print(f"HTTP 404 on page {page} persisted - treating as the end of available pages.", file=sys.stderr)
                page_data = []
                break

            resp.raise_for_status()  # any other 4xx (bad key, bad params) is a real problem - surface it
            page_data = resp.json().get("data", [])
            break
        else:
            print(f"Page {page} failed after {max_retries} attempts. Skipping this page for today.", file=sys.stderr)
            skipped_pages.append(page)
            page += 1
            time.sleep(2)
            if page > 5 and len(skipped_pages) == page - 1:
                raise RuntimeError(
                    f"First {page-1} pages all failed - the API or key is likely "
                    "unavailable rather than momentarily flaky. Aborting this run."
                )
            continue

        if not page_data:
            break
        yield from page_data
        page += 1

    if skipped_pages:
        print(f"Note: {len(skipped_pages)} page(s) skipped after repeated server errors: {skipped_pages}", file=sys.stderr)


# ---------------------------------------------------------------------------
# FLATTEN: one row per card, wide schema, exact column order requested
# ---------------------------------------------------------------------------
def flatten_snapshot(cards, snapshot_date: str, eur_usd_rate: float) -> pd.DataFrame:
    rows = []
    for card in cards:
        set_info = card.get("set") or {}
        tcg = card.get("tcgplayer") or {}
        tcg_prices = tcg.get("prices") or {}
        cm = card.get("cardmarket") or {}
        cm_prices = cm.get("prices") or {}

        # --- pick which variant backs the generic Market/Low/Mid/High columns ---
        primary_key = next((k for k in PRIMARY_VARIANT_PRIORITY if k in tcg_prices), None)
        primary = tcg_prices.get(primary_key) or {} if primary_key else {}

        # --- named-variant single price columns ---
        named_variant_prices = {}
        other_variants = {}
        for variant_key, variant_fields in tcg_prices.items():
            if variant_key == primary_key:
                continue  # already represented in the generic columns
            if variant_key in NAMED_VARIANT_COLUMNS:
                named_variant_prices[NAMED_VARIANT_COLUMNS[variant_key]] = (
                    (variant_fields or {}).get("market")
                )
            else:
                # e.g. "unlimitedHolofoil" - not in the requested column list,
                # preserved here instead of silently dropped
                other_variants[variant_key] = variant_fields

        # --- price_source: which of tcgplayer/cardmarket actually have data ---
        sources = []
        if tcg_prices:
            sources.append("tcgplayer")
        if cm_prices:
            sources.append("cardmarket")
        price_source = ",".join(sources) if sources else None

        # --- purchase URL: prefer tcgplayer's, fall back to cardmarket's ---
        purchase_url = tcg.get("url") or cm.get("url")

        # --- card image: prefer high-res, fall back to the small thumbnail ---
        images = card.get("images") or {}
        image_url = images.get("large") or images.get("small")

        # --- list-type fields -> pipe-joined strings (CSV-safe) ---
        subtypes = card.get("subtypes") or []
        dex_numbers = card.get("nationalPokedexNumbers") or []

        row = {
            "card_id": card.get("id"),
            "name": card.get("name"),
            "variant": primary_key,  # which price group backs Market/Low/Mid/High below
            "set": set_info.get("name"),
            # the card's printed number within its set (e.g. "4" in a
            # 102-card set, or "TG01" for special subsets like Trainer
            # Gallery) - kept as a string since it isn't always purely
            # numeric; sorting logic in the app handles that
            "card_number": card.get("number"),
            "rarity": card.get("rarity"),
            "artist": card.get("artist"),
            "subtype": "|".join(subtypes) if subtypes else None,
            "national_dex_numbers": "|".join(str(n) for n in dex_numbers) if dex_numbers else None,
            "supertype": card.get("supertype"),
            "release_date": set_info.get("releaseDate"),
            "price_source": price_source,
            "purchase_url": purchase_url,
            "image_url": image_url,
            # the source's own "when was this price last refreshed" timestamp -
            # NOT the same as `date` below, which is just the day WE happened
            # to fetch it. A price can sit unchanged for days between the
            # source's own updates even though we pull a snapshot daily.
            "tcgplayer_updated_at": tcg.get("updatedAt"),
            "cardmarket_updated_at": cm.get("updatedAt"),

            "tcgplayer_market": primary.get("market"),
            "tcgplayer_low": primary.get("low"),
            "tcgplayer_mid": primary.get("mid"),
            "tcgplayer_high": primary.get("high"),
            "tcgplayer_holofoil_market": named_variant_prices.get("tcgplayer_holofoil_market"),
            "tcgplayer_reverse_holofoil_market": named_variant_prices.get("tcgplayer_reverse_holofoil_market"),
            "tcgplayer_1st_edition_normal_market": named_variant_prices.get("tcgplayer_1st_edition_normal_market"),
            "tcgplayer_1st_edition_holofoil_market": named_variant_prices.get("tcgplayer_1st_edition_holofoil_market"),
            "other_tcgplayer_variants_json": json.dumps(other_variants) if other_variants else None,

            "cardmarket_average_sell_price_usd": eur_to_usd(cm_prices.get("averageSellPrice"), eur_usd_rate),
            "cardmarket_low_price_usd": eur_to_usd(cm_prices.get("lowPrice"), eur_usd_rate),
            "cardmarket_trend_price_usd": eur_to_usd(cm_prices.get("trendPrice"), eur_usd_rate),
            "cardmarket_reverse_holo_sell_usd": eur_to_usd(cm_prices.get("reverseHoloSell"), eur_usd_rate),
            "cardmarket_avg1_usd": eur_to_usd(cm_prices.get("avg1"), eur_usd_rate),
            "cardmarket_avg7_usd": eur_to_usd(cm_prices.get("avg7"), eur_usd_rate),
            "cardmarket_avg30_usd": eur_to_usd(cm_prices.get("avg30"), eur_usd_rate),
            "cardmarket_reverse_holo_avg1_usd": eur_to_usd(cm_prices.get("reverseHoloAvg1"), eur_usd_rate),
            "cardmarket_reverse_holo_avg7_usd": eur_to_usd(cm_prices.get("reverseHoloAvg7"), eur_usd_rate),
            "cardmarket_reverse_holo_avg30_usd": eur_to_usd(cm_prices.get("reverseHoloAvg30"), eur_usd_rate),

            "eur_usd_rate_used": eur_usd_rate if cm_prices else None,
            "date": snapshot_date,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=COLUMN_ORDER)

    df = pd.DataFrame(rows)
    # enforce exact column order every time, even if some columns ended up
    # all-None for this batch (reindex adds them back rather than shifting
    # everything else out of place)
    df = df.reindex(columns=COLUMN_ORDER)

    # downcast dtypes to keep the growing store memory-efficient
    for col in ("card_id", "name", "variant", "set", "rarity", "artist",
                "subtype", "supertype", "price_source"):
        df[col] = df[col].astype("category")
    for col in df.columns:
        if col.endswith(("_market", "_low", "_mid", "_high", "_usd", "_used")):
            df[col] = pd.to_numeric(df[col], downcast="float")

    return df


# ---------------------------------------------------------------------------
# DAILY FILE PARTITIONING
# ---------------------------------------------------------------------------
# One file per day (data/daily/2026-08-08.csv) instead of one ever-growing
# file. Two problems this solves:
#   1. GitHub rejects any single file over 100MB. A single accumulating file
#      grows without bound and WILL eventually hit that wall (it already
#      has) - a daily file only ever holds one day's worth of rows, so it
#      never grows past roughly the same size no matter how long the
#      dataset has been running.
#   2. The old atomic_append had to read the ENTIRE historical file into
#      memory and rewrite it from scratch every single day just to add one
#      day's rows - that gets slower and heavier forever. Writing a
#      standalone daily file needs no read of prior history at all.
#
# ENCODING NOTE: "utf-8-sig" writes a BOM (byte-order marker) at the start
# of the file. Plain "utf-8" is technically correct, but Excel on Windows
# ignores the file's actual encoding and guesses ANSI/Windows-1252 unless a
# BOM tells it otherwise - that's what produced "PokÃ©mon" instead of
# "Pokémon" earlier. The BOM costs nothing for pandas/Python, which handle
# it transparently, but fixes the display in Excel.
def daily_file_path(data_dir: str, snapshot_date: str) -> str:
    return os.path.join(data_dir, f"{snapshot_date}.csv")


def already_fetched_today(data_dir: str, snapshot_date: str) -> bool:
    return os.path.exists(daily_file_path(data_dir, snapshot_date))


def write_daily_snapshot(new_rows: pd.DataFrame, data_dir: str, snapshot_date: str):
    os.makedirs(data_dir, exist_ok=True)
    final_path = daily_file_path(data_dir, snapshot_date)
    tmp_path = final_path + ".tmp"
    new_rows = new_rows.reindex(columns=COLUMN_ORDER)
    new_rows.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    os.replace(tmp_path, final_path)  # atomic on POSIX and Windows
    return final_path


def load_all_snapshots(data_dir: str) -> pd.DataFrame:
    """
    Reads every daily file and concatenates them into one dataframe -
    the equivalent of what used to be a single pd.read_csv(STORE_PATH)
    call, now spread across many small files. Used by the Streamlit app,
    the LSTM pipeline, and the training notebook.
    """
    paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No daily snapshot files found in {data_dir}")
    frames = [pd.read_csv(p, encoding="utf-8-sig", low_memory=False) for p in paths]
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-key", default=os.environ.get("POKEMONTCG_API_KEY"),
        help="Defaults to the POKEMONTCG_API_KEY env var.",
    )
    parser.add_argument("--data-dir", default="data/daily", help="Directory holding one CSV per day")
    parser.add_argument("--flush-every", type=int, default=500, help="Cards buffered per flatten batch")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit(
            "No API key found. Set POKEMONTCG_API_KEY as an environment "
            "variable, or pass --api-key (less recommended)."
        )

    snapshot_date = date.today().isoformat()

    if already_fetched_today(args.data_dir, snapshot_date):
        print(f"Already have a snapshot for {snapshot_date} in {args.data_dir}. Skipping.")
        return

    eur_usd_rate = get_eur_usd_rate()
    print(f"Using EUR->USD rate: {eur_usd_rate}")

    cards_buffer, dfs = [], []
    total_cards = 0
    for card in fetch_all_cards(args.api_key):
        cards_buffer.append(card)
        total_cards += 1
        if len(cards_buffer) >= args.flush_every:
            dfs.append(flatten_snapshot(cards_buffer, snapshot_date, eur_usd_rate))
            cards_buffer = []
    if cards_buffer:
        dfs.append(flatten_snapshot(cards_buffer, snapshot_date, eur_usd_rate))

    non_empty = [d for d in dfs if not d.empty]
    if not non_empty:
        print("No price rows fetched - check your API key and network connection.")
        return

    new_rows = pd.concat(non_empty, ignore_index=True)
    written_path = write_daily_snapshot(new_rows, args.data_dir, snapshot_date)
    print(f"Fetched {total_cards} cards, wrote {len(new_rows)} rows for {snapshot_date} to {written_path}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# SCHEDULING OPTIONS (pick one)
# ---------------------------------------------------------------------------
# Option A - cron (Mac/Linux):
#   0 6 * * * cd /path/to/project && POKEMONTCG_API_KEY=yourkey python3 build_price_dataset.py --data-dir data/daily >> fetch.log 2>&1
# Option B - Windows Task Scheduler: Daily trigger -> python.exe with the same arguments.
# Option C - GitHub Actions: see build_price_dataset_workflow.yml.
