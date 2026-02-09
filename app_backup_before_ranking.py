import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# إعدادات عامة
# =========================
st.set_page_config(page_title="Trading App (Clean)", layout="wide")

US_SYMBOLS_PATH = Path("data/universe/us_symbols.txt")
SA_SYMBOLS_PATH = Path("data/universe/sa_symbols.txt")

PERIOD_OPTIONS = {
    "1mo": "1mo",
    "3mo": "3mo",
    "6mo": "6mo",
    "1y":  "1y",
    "2y":  "2y",
    "5y":  "5y",
    "max": "max",
}

RISK_PRESETS = {
    "منخفض":  {"atr_mult": 3.0, "swing_lookback": 20},
    "متوسط":  {"atr_mult": 2.2, "swing_lookback": 14},
    "عالي":   {"atr_mult": 1.6, "swing_lookback": 10},
}

# =========================
# Helpers
# =========================
def load_symbols(path: Path):
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = (ln or "").strip()
        if not s:
            continue
        out.append(s.upper())
    # unique preserving order
    seen = set()
    clean = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        clean.append(s)
    return clean

def fmt_symbol(symbol: str, market: str) -> str:
    s = (symbol or "").strip().upper()
    if market == "SA":
        if s.isdigit():
            return f"{s}.SR"
        if s.endswith(".SR"):
            return s
    return s

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(method="bfill").fillna(50)

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean().fillna(method="bfill")

def swing_low(df: pd.DataFrame, lookback: int = 14):
    if len(df) < lookback:
        return float(df["Low"].min())
    return float(df["Low"].tail(lookback).min())

@st.cache_data(show_spinner=False)
def fetch_history(symbol: str, period: str):
    t = yf.Ticker(symbol)
    df = t.history(period=period, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna()
    return df

def calc_plan(df: pd.DataFrame, risk_mode: str):
    if df is None or df.empty or len(df) < 40:
        return None

    close = df["Close"]
    last = float(close.iloc[-1])

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    rsi14 = rsi(close, 14)
    atr14 = atr(df, 14)

    last_ema20 = float(ema20.iloc[-1])
    last_ema50 = float(ema50.iloc[-1])
    last_rsi = float(rsi14.iloc[-1])
    last_atr = float(atr14.iloc[-1])

    p = RISK_PRESETS.get(risk_mode, RISK_PRESETS["متوسط"])
    atr_mult = float(p["atr_mult"])
    lookback = int(p["swing_lookback"])

    trend_up = (last_ema20 > last_ema50) and (last > last_ema20)
    trend_down = (last_ema20 < last_ema50) and (last < last_ema20)
    trend = "صاعد" if trend_up else ("هابط" if trend_down else "متذبذب")

    reasons = []
    suitable = False

    if trend_up:
        reasons.append("الاتجاه صاعد (EMA20 فوق EMA50 والسعر فوق EMA20)")
        if 40 <= last_rsi <= 70:
            suitable = True
            reasons.append("RSI ضمن نطاق مريح (40–70)")
        else:
            reasons.append(f"RSI خارج النطاق المفضل (حاليًا {last_rsi:.1f})")
    elif trend_down:
        reasons.append("الاتجاه هابط (حذر)")
        reasons.append(f"RSI الحالي {last_rsi:.1f}")
    else:
        reasons.append("الاتجاه غير واضح (تذبذب)")
        reasons.append(f"RSI الحالي {last_rsi:.1f}")

    # Entry: pullback near EMA20 in uptrend, else last
    if trend_up:
        entry = round(min(last, last_ema20 * 1.01), 2)
    else:
        entry = round(last, 2)

    # Stop: under swing low with ATR cushion
    sw_low = swing_low(df, lookback=lookback)
    stop = sw_low - (last_atr * atr_mult)
    stop = round(max(0.01, stop), 2)

    risk = max(0.01, entry - stop)
    t1 = round(entry + 1.0 * risk, 2)
    t2 = round(entry + 2.0 * risk, 2)
    t3 = round(entry + 3.0 * risk, 2)

    rr1 = 1.0
    rr2 = 2.0
    rr3 = 3.0

    return {
        "symbol_last": round(last, 2),
        "ema20": round(last_ema20, 2),
        "ema50": round(last_ema50, 2),
        "rsi": round(last_rsi, 1),
        "atr": round(last_atr, 2),
        "trend": trend,
        "suitable": suitable,
        "reasons": reasons,
        "entry": entry,
        "stop": stop,
        "risk_per_share": round(risk, 2),
        "targets": [(t1, rr1), (t2, rr2), (t3, rr3)],
        "risk_mode": risk_mode,
        "atr_mult": atr_mult,
        "lookback": lookback,
    }

def market_box(title: str, market_code: str, symbols: list[str], limit_buttons: int = 120):
    st.markdown(f"### {title}")

    if not symbols:
        missing = SA_SYMBOLS_PATH if market_code == "SA" else US_SYMBOLS_PATH
        st.warning(f"ملف القائمة غير موجود: {missing}")
        return

    q = st.text_input("بحث سريع", key=f"q_{market_code}", placeholder="اكتب جزء من الرمز…")
    q = (q or "").strip().upper()

    filtered = symbols
    if q:
        filtered = [s for s in symbols if q in s]

    st.caption(f"النتائج: {len(filtered)}" + (f" (نعرض أول {limit_buttons})" if len(filtered) > limit_buttons else ""))

    with st.container(height=360):
        for s in filtered[:limit_buttons]:
            if st.button(s, key=f"pick_{market_code}_{s}", use_container_width=True):
                st.session_state.selected_market = market_code
                st.session_state.selected_symbol = s
                st.rerun()

# =========================
# Load universes
# =========================
us_symbols = load_symbols(US_SYMBOLS_PATH)
sa_symbols = load_symbols(SA_SYMBOLS_PATH)

# =========================
# UI
# =========================
st.title("📌 الأسواق + تحليل سريع")

# Controls
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    period_label = st.selectbox("المدة", list(PERIOD_OPTIONS.keys()), index=2)
with c2:
    risk_mode = st.selectbox("المخاطرة", list(RISK_PRESETS.keys()), index=1)
with c3:
    st.caption("اختر سهم من الصناديق باليسار 👇")

st.divider()

left, right = st.columns([1.05, 1.6], vertical_alignment="top")

with left:
    st.subheader("🗂️ القوائم")
    box_us, box_sa = st.columns(2)
    with box_us:
        market_box("🇺🇸 الأمريكي", "US", us_symbols)
    with box_sa:
        market_box("🇸🇦 السعودي", "SA", sa_symbols)

with right:
    st.subheader("🔎 التحليل")

    sel_market = st.session_state.get("selected_market")
    sel_symbol_raw = st.session_state.get("selected_symbol")

    if not sel_symbol_raw:
        st.info("اضغط على أي سهم عشان يطلع التحليل هنا.")
        st.stop()

    symbol = fmt_symbol(sel_symbol_raw, sel_market or "US")

    st.markdown(f"### السهم: *{symbol}*")

    with st.spinner("جاري جلب البيانات..."):
        df = fetch_history(symbol, PERIOD_OPTIONS[period_label])

    if df.empty:
        st.error("❌ ما لقيت بيانات لهذا السهم (ممكن الرمز خطأ أو لا يوجد بيانات في Yahoo).")
        st.stop()

    plan = calc_plan(df, risk_mode=risk_mode)
    if not plan:
        st.error("❌ البيانات غير كافية للتحليل (جرّب مدة أطول).")
        st.stop()

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("السعر الحالي", f"{plan['symbol_last']}")
    k2.metric("الاتجاه", plan["trend"])
    k3.metric("RSI", f"{plan['rsi']}")
    k4.metric("ATR", f"{plan['atr']}")

    st.divider()

    # Decision box
    if plan["suitable"]:
        st.success("✅ مناسب مبدئيًا للدخول حسب القواعد الحالية.")
    else:
        st.warning("⚠️ غير مناسب مبدئيًا (أو يحتاج تأكيد).")

    st.markdown("*الأسباب:*")
    for r in plan["reasons"]:
        st.write("•", r)

    st.divider()

    # Trade Plan (clean cards)
    st.markdown("## 🎯 خطة الدخول (مقترحة)")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("دخول", f"{plan['entry']}")
    p2.metric("وقف", f"{plan['stop']}")
    p3.metric("مخاطرة/سهم", f"{plan['risk_per_share']}")
    p4.metric("الوضع", plan["risk_mode"])

    st.markdown("### 🎯 الأهداف")
    tcols = st.columns(3)
    for i, (t, rr) in enumerate(plan["targets"]):
        tcols[i].metric(f"هدف {i+1} (R{rr:.0f})", f"{t}")

    st.caption(f"ملاحظة: الوقف مبني على قاع {plan['lookback']} يوم + ATR×{plan['atr_mult']}")

    # Optional: last prices
    st.divider()
    show_last = st.checkbox("📌 عرض آخر الأسعار (مختصر)", value=False)
    if show_last:
        days = st.slider("كم يوم؟", min_value=5, max_value=30, value=10)
        tail = df.tail(days).copy()
        tail.index = tail.index.astype(str)
        st.dataframe(tail[["Open", "High", "Low", "Close", "Volume"]], use_container_width=True)
