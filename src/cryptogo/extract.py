"""Pull hourly CSVs out of the Kraken OHLCVT zip archives."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "kraken"
FULL_ZIP = DATA / "Kraken_OHLCVT_Full.zip"
Q3_ZIP = DATA / "Kraken_OHLCVT_2026Q3.zip"
HOURLY = DATA / "hourly"


def extract_hourly(zip_path: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    kept = 0
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if Path(name).name.endswith("_60.csv")]
        if not names:
            sample = archive.namelist()[:30]
            raise RuntimeError(f"no hourly CSVs in {zip_path.name}; sample entries: {sample}")
        for name in names:
            target = dest / Path(name).name
            if target.exists() and target.stat().st_size > 0:
                kept += 1
                continue
            with archive.open(name) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            kept += 1
            if kept % 200 == 0:
                print(f"  {zip_path.name}: {kept} hourly files", flush=True)
    print(f"{zip_path.name}: {kept} hourly files -> {dest}", flush=True)
    return kept


def ensure_hourly() -> Path:
    full_dir = HOURLY / "full"
    q3_dir = HOURLY / "q3"
    if not any(full_dir.glob("*_60.csv")):
        extract_hourly(FULL_ZIP, full_dir)
    if Q3_ZIP.exists() and not any(q3_dir.glob("*_60.csv")):
        extract_hourly(Q3_ZIP, q3_dir)
    elif not Q3_ZIP.exists():
        print("Q3 2026 zip is absent; the panel ends where the full archive ends.", flush=True)
    return HOURLY
