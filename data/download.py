"""
data/download.py
----------------
Utility script to download / update raw datasets for the Famine Trader project:
1. NOAA Oceanic Niño Index (weather/climate anomalies)
2. FAO Food Price Index (monthly global cereal price indices)
3. Chicago Wheat Futures (ZW=F) via yfinance (if installed) or direct CSV URL

Usage:
    python data/download.py --output data/raw
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NOAA_ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
FAO_FFPI_URL = "https://www.fao.org/fileadmin/templates/worldfood/Reports_and_docs/Food_price_indices_data_csv.csv"


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file using standard library urllib."""
    try:
        logger.info("Downloading %s -> %s", url, dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as f:
            f.write(resp.read())
        logger.info("Successfully downloaded %s (size: %d bytes)", dest_path.name, dest_path.stat().st_size)
        return True
    except Exception as exc:
        logger.warning("Could not download %s: %s", url, exc)
        return False


def download_wheat_futures(dest_path: Path, start_date: str = "2005-01-01") -> bool:
    """Download historical wheat futures (ZW=F) using yfinance if available."""
    try:
        import yfinance as yf
        logger.info("Fetching Chicago Wheat Futures (ZW=F) via yfinance...")
        df = yf.download("ZW=F", start=start_date, progress=False)
        if df.empty:
            logger.warning("No data returned for ZW=F.")
            return False
        # Handle MultiIndex columns in recent yfinance versions
        if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
            close_col = df["Close"]["ZW=F"] if "ZW=F" in df["Close"] else df["Close"].iloc[:, 0]
        else:
            close_col = df["Close"]

        out = close_col.dropna().reset_index()
        out.columns = ["Date", "Close"]
        out.to_csv(dest_path, index=False)
        logger.info("Saved %d rows of wheat futures to %s", len(out), dest_path)
        return True
    except ImportError:
        logger.info("yfinance not installed. To download ZW=F automatically, install yfinance (`pip install yfinance`).")
        return False
    except Exception as exc:
        logger.warning("Error fetching wheat futures: %s", exc)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download raw datasets for Famine Trader")
    parser.add_argument("--output", default="data/raw", help="Target folder for raw datasets")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. NOAA ONI
    oni_path = out_dir / "oni.txt"
    if not oni_path.exists():
        download_file(NOAA_ONI_URL, oni_path)
    else:
        logger.info("NOAA ONI dataset already present at %s", oni_path)

    # 2. FAO FFPI
    fao_path = out_dir / "fao_ffpi.csv"
    if not fao_path.exists():
        download_file(FAO_FFPI_URL, fao_path)
    else:
        logger.info("FAO FFPI dataset already present at %s", fao_path)

    # 3. Wheat Futures (ZW=F)
    wheat_path = out_dir / "wheat_zw.csv"
    if not wheat_path.exists():
        download_wheat_futures(wheat_path)
    else:
        logger.info("Wheat futures dataset already present at %s", wheat_path)


if __name__ == "__main__":
    main()
