import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None


# =============================
# Paths / Universe
# =============================
DATA_DIR = Path("data/universe")
US_SYMBOLS_FILE = DATA_DIR / "us_symbols.txt"
SA_SYMBOLS_FILE = DATA_DIR / "sa_symbols.txt"   # تاسي فقط، رموز أرقام + .SR

DATA_DIR.mkdir(parents=True, exist_ok=True)


def _read_symbols(path: Path) -> list[str]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s.upper())
    return sorted(list(dict.fromkeys(out)))


def load_universe(market: str) -> list[str]:
    if market == "US":
        return _read_symbols(US_SYMBOLS_FILE)
    if market == "SA":
        syms = _read_symbols(SA_SYMBOLS_FILE)
        fixed = []
        for s in syms:
            s = s.upper()
            if s.isdigit():
                fixed.append(f"{s}.SR")
            elif s.endswith(".SR"):
                fixed.append(s)
        return sorted(list(dict.fromkeys(fixed)))
    return []


# =============================
# Indicators
# =============================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, pd.NA))
    out = 100 - (100 / (1 + rs))
    return out.fillna(method="bfill").fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


@dataclass
class Plan:
    entry: float
    stop: float
    targets: list[float]
    risk_label: str
    rr: float


def risk_label_from_atr_pct(atr_pct: float) -> str:
    if atr_pct < 2.0:
        return "منخفض"
    if atr_pct < 4.0:
        return "متوسط"
    return "مرتفع"


def build_plan(price: float, atr_v: float) -> Plan:
    entry = float(price)
    stop = float(max(0.01, entry - 2.0 * atr_v))
    targets = [entry + 2.0 * atr_v, entry + 3.0 * atr_v, entry + 4.0 * atr_v]
    atr_pct = (atr_v / entry) * 100 if entry else 0.0
    risk_label = risk_label_from_atr_pct(atr_pct)

    risk = max(0.01, entry - stop)
    reward = max(0.01, targets[0] - entry)
    rr = reward / risk if risk else 0.0

    return Plan(entry=entry, stop=stop, targets=[float(x) for x in targets], risk_label=risk_label, rr=float(rr))


def score_opportunity(df: pd.DataFrame) -> tuple[float, dict]:
    close = df["Close"]
    e20 = ema(close, 20)
    e50 = ema(close, 50)
    r = rsi(close, 14)

    last = df.iloc[-1]
    price = float(last["Close"])
    e20v = float(e20.iloc[-1])
    e50v = float(e50.iloc[-1])
    rv = float(r.iloc[-1])

    trend = 0.0
    if price > e20v:
        trend += 2.0
    if e20v > e50v:
        trend += 2.0
    if price > e50v:
        trend += 1.0

    rsi_pts = 0.0
    if 45 <= rv <= 65:
        rsi_pts += 3.0
    elif 35 <= rv < 45 or 65 < rv <= 75:
        rsi_pts += 1.5

    hh20 = float(df["High"].iloc[-21:-1].max()) if len(df) >= 21 else float(df["High"].max())
    breakout = 3.0 if price > hh20 else 0.0

    score = trend + rsi_pts + breakout
    info = {"price": price, "ema20": e20v, "ema50": e50v, "rsi": rv, "breakout": breakout > 0, "score": score}
    return float(score), info


# =============================
# Data fetch
# =============================
@st.cache_data(show_spinner=False, ttl=60 * 10)
def fetch_history(symbol: str, period: str) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.dropna()


@st.cache_data(show_spinner=False, ttl=60 * 10)
def fetch_history_batch(symbols: list[str], period: str) -> dict[str, pd.DataFrame]:
    """
    تحميل دفعة وحدة عشان ما يصير Too many open files
    يرجع dict: symbol -> df
    """
    if yf is None or not symbols:
        return {}
    tickers = " ".join(symbols)
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    out: dict[str, pd.DataFrame] = {}

    if raw is None or raw.empty:
        return out

    # حالة سهم واحد: أعمدة عادية
    if not isinstance(raw.columns, pd.MultiIndex):
        df = raw.dropna()
        # ما نعرف الرمز الحقيقي من raw هنا، نستخدم أول رمز
        out[symbols[0]] = df
        return out

    # حالة عدة أسهم: MultiIndex (ticker, field)
    for s in symbols:
        if s not in raw.columns.get_level_values(0):
            continue
        df = raw[s].dropna()
        if df is not None and not df.empty:
            out[s] = df
    return out


def fmt_price(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:,.2f}"


# =============================
# UI
# =============================
st.set_page_config(page_title="Trading App (Clean)", layout="wide")

st.title("📌 الأسواق")
st.caption("اختر السوق ثم السهم، ويطلع التحليل والخطة.")

c1, c2 = st.columns([1, 9])
with c1:
    if st.button("🔄 تحديث"):
        st.cache_data.clear()
        for k in list(st.session_state.keys()):
            if k.startswith("sel_") or k.startswith("q_") or k.startswith("best_"):
                del st.session_state[k]
        st.rerun()

left, right = st.columns([1.05, 1.95], gap="large")

with left:
    st.subheader("🧰 القوائم")
    box_us = st.container(border=True)
    box_sa = st.container(border=True)

    st.markdown("---")
    period = st.selectbox("اختر المدة", ["1mo", "3mo", "6mo", "1y", "2y"], index=2, key="sel_period")

    us_all = load_universe("US")
    sa_all = load_universe("SA")

    with box_us:
        st.markdown("### 🇺🇸 السوق الأمريكي (ناسداك)")
        if not us_all:
            st.warning("جهّز: data/universe/us_symbols.txt")
        q_us = st.text_input("بحث سريع", value=st.session_state.get("q_us", ""), key="q_us")
        us_filtered = [s for s in us_all if q_us.strip().upper() in s] if q_us.strip() else us_all
        page_size_us = st.selectbox("عدد الأسهم المعروضة هنا", [50, 100, 200, 500], index=1, key="sel_us_page")
        us_show = us_filtered[:page_size_us]
        st.selectbox("اختر سهم أمريكي", us_show if us_show else ["—"], index=0, key="sel_us_symbol")

    with box_sa:
        st.markdown("### 🇸🇦 السوق السعودي (تاسي)")
        if not sa_all:
            st.warning("جهّز: data/universe/sa_symbols.txt (مثل 1180.SR)")
        q_sa = st.text_input("بحث سريع", value=st.session_state.get("q_sa", ""), key="q_sa")
        qv = q_sa.strip().upper()
        sa_filtered = [s for s in sa_all if qv in s] if qv else sa_all
        page_size_sa = st.selectbox("عدد الأسهم المعروضة هنا", [50, 100, 200, 500], index=1, key="sel_sa_page")
        sa_show = sa_filtered[:page_size_sa]
        st.selectbox("اختر سهم سعودي", sa_show if sa_show else ["—"], index=0, key="sel_sa_symbol")

    st.markdown("---")
    active_market = st.radio(
        "السوق المعتمد للتحليل",
        ["US", "SA"],
        format_func=lambda x: "🇺🇸 أمريكي" if x == "US" else "🇸🇦 سعودي",
        horizontal=True,
        key="sel_market",
    )

    chosen_symbol = st.session_state.get("sel_us_symbol", "—") if active_market == "US" else st.session_state.get("sel_sa_symbol", "—")
    st.caption(f"السهم المختار الآن: *{chosen_symbol}*")

with right:
    st.subheader("📊 التحليل")

    symbol = chosen_symbol
    if symbol in (None, "", "—"):
        st.info("اختر سهم من القائمة يسار.")
        st.stop()

    if yf is None:
        st.error("ثبّت yfinance: pip install yfinance")
        st.stop()

    df = fetch_history(symbol, period)
    if df.empty:
        st.error(f"ما قدرت أجيب بيانات للسهم: {symbol}")
        st.stop()

    close = df["Close"]
    e20 = ema(close, 20)
    e50 = ema(close, 50)
    r = rsi(close, 14)
    a = atr(df, 14)

    last = df.iloc[-1]
    price = float(last["Close"])
    rsi_v = float(r.iloc[-1])
    atr_v = float(a.iloc[-1])

    reasons = []
    ok = True

    if price < float(e20.iloc[-1]):
        ok = False
        reasons.append("السعر تحت EMA20 (ضعف قصير المدى).")
    if float(e20.iloc[-1]) < float(e50.iloc[-1]):
        ok = False
        reasons.append("EMA20 تحت EMA50 (الاتجاه غير مؤكد).")
    if rsi_v < 40:
        ok = False
        reasons.append(f"RSI منخفض ({rsi_v:.1f}) — ضعف.")
    if rsi_v > 75:
        ok = False
        reasons.append(f"RSI عالي ({rsi_v:.1f}) — تشبع شراء.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("السعر الحالي", fmt_price(price))
    m2.metric("الاتجاه", "صاعد" if (price > float(e20.iloc[-1]) and float(e20.iloc[-1]) > float(e50.iloc[-1])) else "متذبذب")
    m3.metric("RSI", f"{rsi_v:.1f}")
    m4.metric("ATR", f"{atr_v:.2f}")

    if ok:
        st.success("✅ مناسب مبدئيًا للدخول حسب القواعد الحالية.")
    else:
        st.warning("⚠️ غير مناسب مبدئيًا حسب القواعد الحالية (ويحتاج تأكيد).")

    if reasons:
        st.markdown("*الأسباب:*")
        for x in reasons:
            st.write(f"• {x}")

    st.markdown("### 🎯 خطة الدخول (مقترحة)")
    plan = build_plan(price, atr_v)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("دخول (Entry)", fmt_price(plan.entry))
    p2.metric("وقف (Stop)", fmt_price(plan.stop))
    p3.metric("مخاطرة/سهم", fmt_price(plan.entry - plan.stop))
    p4.metric("الوضع", plan.risk_label)

    t1, t2, t3 = st.columns(3)
    t1.metric("هدف 1 (R1)", fmt_price(plan.targets[0]))
    t2.metric("هدف 2 (R2)", fmt_price(plan.targets[1]))
    t3.metric("هدف 3 (R3)", fmt_price(plan.targets[2]))

    st.caption(f"الخطة مبنية على ATR (14 يوم) — RR تقريبي: {plan.rr:.2f}")

    st.markdown("---")
    show_last = st.checkbox("📌 عرض آخر الأسعار (مختصر)", value=st.session_state.get("show_last", False), key="show_last")
    if show_last:
        tail = df.tail(12).copy()
        tail = tail.reset_index().rename(columns={"Date": "التاريخ"})
        tail = tail[["التاريخ", "Open", "High", "Low", "Close", "Volume"]]
        st.dataframe(tail, width="stretch")


# =============================
# Best Opportunities (Batch fix)
# =============================
st.markdown("---")
st.subheader("🏆 أفضل الفرص (اختياري)")

bwrap = st.container(border=True)
with bwrap:
    best_market = st.selectbox("اختر السوق للفحص", ["SA", "US"], index=0, key="best_market",
                              format_func=lambda x: "🇸🇦 تاسي" if x == "SA" else "🇺🇸 ناسداك")
    show_top = st.selectbox("اعرض الأفضل", [5, 10, 20, 50], index=1, key="best_top")
    scan_n = st.selectbox("كم سهم نفحص من القائمة؟", [50, 100, 200, 500], index=1, key="best_scan")
    run = st.button("🚀 افحص أفضل الفرص", key="best_run")

    if run:
        all_syms = load_universe(best_market)
        if not all_syms:
            st.error("القائمة فاضية — جهّز ملفات السوق أول.")
        else:
            syms = all_syms[: int(scan_n)]

            st.info("جالس أحمل البيانات دفعة وحدة…")
            prog = st.progress(0)

            # تحميل Batch واحد
            data_map = fetch_history_batch(syms, st.session_state["sel_period"])
            prog.progress(0.5)

            rows = []
            for i, s in enumerate(syms, start=1):
                d = data_map.get(s)
                if d is None or d.empty or len(d) < 60:
                    continue
                sc, info = score_opportunity(d)
                rows.append({
                    "الرمز": s,
                    "Score": round(sc, 2),
                    "الاتجاه": "صاعد" if (info["price"] > info["ema20"] and info["ema20"] > info["ema50"]) else "متذبذب",
                    "RSI": round(info["rsi"], 1),
                    "Breakout": "✅" if info["breakout"] else "—",
                    "السعر": round(info["price"], 2),
                })
            prog.progress(1.0)

            if not rows:
                st.warning("ما طلعت نتائج كفاية (جرّب زود scan أو غيّر المدة).")
            else:
                out = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
                st.success(f"تم ترتيب {len(out)} سهم ✅")
                st.dataframe(out.head(int(show_top)), width="stretch")
                st.caption("الترتيب يعتمد على: Trend + RSI + Breakout (مبدئي فقط).")
