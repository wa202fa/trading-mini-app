import json
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf


# =========================
# إعدادات عامة
# =========================
st.set_page_config(page_title="Trading App (Clean)", layout="wide")


# =========================
# تحميل قوائم الأسهم
# =========================
DATA_DIR = Path("data/universe")
US_SYMBOLS_TXT = DATA_DIR / "us_symbols.txt"
SA_SYMBOLS_TXT = DATA_DIR / "sa_symbols.txt"
US_ALL_JSON = DATA_DIR / "us_all.json"
SA_ALL_JSON = DATA_DIR / "sa_all.json"


def _read_symbols_txt(p: Path) -> list[str]:
    if not p.exists():
        return []
    rows = []
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = (ln or "").strip()
        if not s:
            continue
        rows.append(s.upper())
    # unique preserving order
    seen = set()
    out = []
    for s in rows:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _read_symbols_from_json(p: Path, market: str) -> list[str]:
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8", errors="ignore") or "[]")
    except Exception:
        return []
    out = []
    for r in data if isinstance(data, list) else []:
        sym = ""
        if isinstance(r, dict):
            sym = str(r.get("symbol", "")).strip()
        if not sym:
            continue
        out.append(sym.upper())
    return out


@st.cache_data(show_spinner=False)
def load_universe_symbols(market: str) -> list[str]:
    market = market.upper()
    if market == "US":
        syms = _read_symbols_txt(US_SYMBOLS_TXT)
        if not syms:
            syms = _read_symbols_from_json(US_ALL_JSON, "US")
        return syms
    if market == "SA":
        syms = _read_symbols_txt(SA_SYMBOLS_TXT)
        if not syms:
            syms = _read_symbols_from_json(SA_ALL_JSON, "SA")
        return syms
    return []


def fmt_symbol(sym: str, market: str) -> str:
    s = (sym or "").strip().upper()
    if not s:
        return ""
    if market == "SA":
        # لو كتب رقم مثل 2222 نخليه 2222.SR
        if s.isdigit():
            return f"{s}.SR"
        # لو كتبها جاهزة
        if s.endswith(".SR"):
            return s
    return s


# =========================
# سلة المتابعة (State)
# =========================
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []  # list[str]


def add_to_watchlist(sym: str):
    sym = sym.strip().upper()
    if not sym:
        return
    if sym not in st.session_state.watchlist:
        st.session_state.watchlist.append(sym)


def remove_from_watchlist(sym: str):
    sym = sym.strip().upper()
    st.session_state.watchlist = [x for x in st.session_state.watchlist if x != sym]


# =========================
# واجهة
# =========================
st.title("🔎 فحص قائمة المتابعة — نسخة نظيفة")

with st.sidebar:
    st.header("📌 القوائم")

    market_label = st.selectbox("السوق", ["🇺🇸 أمريكا", "🇸🇦 السعودية"], index=0)
    market = "US" if "أمريكا" in market_label else "SA"

    period = st.selectbox("اختر المدة", ["1mo", "3mo", "6mo", "1y"], index=2)
    top_n = st.selectbox("كم سهم نعرض في الترتيب", [20, 15, 10, 5, 3], index=1)

    st.divider()
    st.subheader("➕ إضافة سهم للسلة")

    universe = load_universe_symbols(market)

    if universe:
        picked = st.selectbox("اختر سهم من القائمة", universe, index=0)
    else:
        picked = None
        st.warning("⚠️ ملف قوائم الأسهم غير موجود للسوق هذا. (بنضبطه بالخطوة الجاية)")

    manual = st.text_input("أو اكتب (رمز/رقم)", value="", placeholder="مثال: AAPL أو 2222")

    colA, colB = st.columns(2)
    if colA.button("➕ أضف للسلة", use_container_width=True):
        sym = fmt_symbol(manual if manual.strip() else (picked or ""), market)
        add_to_watchlist(sym)

    if colB.button("🗑️ مسح السلة", use_container_width=True):
        st.session_state.watchlist = []

    st.divider()
    st.subheader("🧺 سلة المتابعة")
    if st.session_state.watchlist:
        for s in st.session_state.watchlist:
            c1, c2 = st.columns([3, 1])
            c1.write(f"• {s}")
            if c2.button("✖", key=f"rm_{s}"):
                remove_from_watchlist(s)
                st.rerun()
    else:
        st.caption("فاضية")

    run_scan = st.button("🚀 افحص السلة", use_container_width=True)


# =========================
# الفحص
# =========================
def fetch_last_change(symbol: str, period: str):
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, auto_adjust=False)
        if df is None or df.empty:
            return None, None
        close = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else None
        chg = ((close - prev) / prev * 100.0) if (prev and prev != 0) else None
        return close, chg
    except Exception:
        return None, None


if run_scan:
    wl = st.session_state.watchlist[:]
    if not wl:
        st.info("أضف أسهم للسلة أولاً.")
    else:
        st.subheader("📊 النتائج")
        rows = []
        for sym in wl:
            price, chg = fetch_last_change(sym, period)
            rows.append(
                {
                    "الرمز": sym,
                    "السعر الأخير": None if price is None else round(price, 2),
                    "% التغير": None if chg is None else round(chg, 2),
                    "الحالة": "✅ تم التحليل" if price is not None else "❌ لا توجد بيانات",
                }
            )
        out = pd.DataFrame(rows)

        # ترتيب: الأعلى تغيير أولاً (نزلي)
        out_sorted = out.sort_values(by="% التغير", ascending=False, na_position="last")

        st.dataframe(out_sorted, use_container_width=True)

        st.subheader("🏆 الأعلى تغيير")
        st.dataframe(out_sorted.head(int(top_n)), use_container_width=True)
