import json, re, ssl
from pathlib import Path
from urllib.request import urlopen

OUT_DIR = Path("data/universe")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED  = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

def _read_pipe_file(url: str):
    ctx = ssl.create_default_context()
    raw = urlopen(url, timeout=30, context=ctx).read().decode("utf-8", errors="ignore").splitlines()
    rows = []
    for line in raw:
        if not line or line.startswith("File Creation Time") or line.startswith("Symbol|") or line.startswith("ACT Symbol|"):
            continue
        if line.startswith("Total Records"):
            break
        parts = line.split("|")
        rows.append(parts)
    return rows

def build_us_universe():
    out = []
    seen = set()

    # nasdaqlisted
    for p in _read_pipe_file(NASDAQ_LISTED):
        if len(p) < 8:
            continue
        sym = p[0].strip()
        name = p[1].strip()
        test = p[3].strip()
        if test == "Y":
            continue
        if not re.match(r"^[A-Z0-9.\-]+$", sym):
            continue
        if sym in seen:
            continue
        seen.add(sym)
        out.append({"symbol": sym, "name": name, "market": "US"})

    # otherlisted
    for p in _read_pipe_file(OTHER_LISTED):
        if len(p) < 8:
            continue
        sym = p[0].strip()
        name = p[1].strip()
        test = p[6].strip()
        if test == "Y":
            continue
        if not re.match(r"^[A-Z0-9.\-]+$", sym):
            continue
        if sym in seen:
            continue
        seen.add(sym)
        out.append({"symbol": sym, "name": name, "market": "US"})

    return out

def build_sa_universe_from_file():
    p = OUT_DIR / "sa_symbols.txt"
    if not p.exists():
        return []
    items = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        items.append({"symbol": s, "name": s, "market": "SA"})
    return items

def main():
    us = build_us_universe()
    (OUT_DIR / "us_all.json").write_text(json.dumps(us, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "us_symbols.txt").write_text("\n".join([x["symbol"] for x in us]), encoding="utf-8")
    print(f"✅ US universe saved: {len(us)} symbols -> data/universe/us_all.json")

    sa = build_sa_universe_from_file()
    (OUT_DIR / "sa_all.json").write_text(json.dumps(sa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ SA universe saved: {len(sa)} symbols -> data/universe/sa_all.json")

if _name_ == "_main_":
    main()
