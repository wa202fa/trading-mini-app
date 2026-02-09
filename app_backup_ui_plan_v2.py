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
        s = ln.strip()
        if not s:
            continue
        out.append(s.upper())
    # remove duplicates preserving order
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
        # allow user list to include either 2222 or 2222.SR
        if s.isdigit():
            return f"{s}.SR"
        if not s.endswith(".SR") and s.replace(".", "").isdigit():
            # just in case
            return f"{s}.SR"
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
    return out.fillna(method="bfill")

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean().fillna(method="bfill")

def swing_low(df: pd.DataFrame, lookback: int = 14):
    if len(df) < lookback:
        return float(df["Low"].min())
    return float(df["Low"].tail(lookback).min())

def calc_plan(df: pd.DataFrame, risk_mode: str):
    if df is None or df.empty or len(df) < 30:
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

    # Trend logic (simple & robust)
    trend_up = (last_ema20 > last_ema50) and (last > last_ema20)
    trend_down = (last_ema20 < last_ema50) and (last < last_ema20)

    # Basic "suitable" rule
    suitable = False
    reason = []

    if trend_up:
        reason.append("الاتجاه صاعد (EMA20 فوق EMA50 والسعر فوق EMA20)")
        if 40 <= last_rsi <= 70:
            suitable = True
            reason.append("RSI ضمن نطاق مريح (40-70)")
        else:
            reason.append(f"RSI خارج النطاق المفضل (حاليًا {last_rsi:.1f})")
    elif trend_down:
        reason.append("الاتجاه هابط (حذر)")
        reason.append(f"RSI الحالي {last_rsi:.1f}")
    else:
        reason.append("الاتجاه غير واضح (تذبذب)")
        reason.append(f"RSI الحالي {last_rsi:.1f}")

    # Entry: in uptrend, prefer pullback near EMA20; otherwise last
    if trend_up:
        entry = round(min(last, last_ema20 * 1.01), 2)  # قريب من EMA20
    else:
        entry = round(last, 2)

    # Stop: under swing low - ATR cushion
    sw_low = swing_low(df, lookback=lookback)
    stop = sw_low - (last_atr * atr_mult)
    stop = round(max(0.01, stop), 2)

    # Targets based on R (risk per share)
    risk = max(0.01, entry - stop)
    t1 = round(entry + 1.0 * risk, 2)
    t2 = round(entry + 2.0 * risk, 2)
    t3 = round(entry + 3.0 * risk, 2)

    return {
        "last": round(last, 2),
        "ema20": round(last_ema20, 2),
        "ema50": round(last_ema50, 2),
        "rsi": round(last_rsi, 1),
        "atr": round(last_atr, 2),
        "trend": "صاعد" if trend_up else ("هابط" if trend_down else "متذبذب"),
        "suitable": suitable,
        "reasons": reason,
        "entry": entry,
        "stop": stop,
        "targets": [t1, t2, t3],
        "lookback": lookback,
        "atr_mult": atr_mult,
    }

@st.cache_data(show_spinner=False)
def fetch_history(symbol: str, period: str):
    t = yf.Ticker(symbol)
    df = t.history(period=period, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # Clean
    df = df.dropna()
    return df

def market_box(title: str, market_code: str, symbols: list[str], limit_buttons: int = 120):
    st.markdown(f"### {title}")
    if not symbols:
        st.warning(f"ملف قائمة الأسهم غير موجود: {SA_SYMBOLS_PATH if market_code=='SA' else US_SYMBOLS_PATH}")
        return

    q = st.text_input("ابحث (رمز/اسم مختصر)", key=f"q_{market_code}", placeholder="مثال: AAPL أو 2222")
    q = (q or "").strip().upper()

    filtered = symbols
    if q:
        filtered = [s for s in symbols if q in s]

    # كي لا يهنّق الواجهة إذا القائمة ضخمة
    st.caption(f"عدد النتائج: {len(filtered)}" + (f" (نعرض أول {limit_buttons})" if len(filtered) > limit_buttons else ""))

    with st.container(height=360):
        shown = filtered[:limit_buttons]
        for s in shown:
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
# UI Layout
# =========================
st.title("🔎 فحص قائمة المتابعة — نسخة نظيفة")

# Controls row
c1, c2, c3 = st.columns([1,1,1], vertical_alignment="top")
with c1:
    period = st.selectbox("اختر المدة", list(PERIOD_OPTIONS.keys()), index=2)
with c2:
    risk_mode = st.selectbox("وضع المخاطرة", list(RISK_PRESETS.keys()), index=1)
with c3:
    st.write("")
    st.caption("اختر سهم من أحد السوقين بالأسفل 👇")

st.divider()

# Market boxes + Analysis panel
left, right = st.columns([1.05, 1.4], vertical_alignment="top")

with left:
    st.subheader("📌 الأسواق")
    box_us, box_sa = st.columns(2)
    with box_us:
        market_box("🇺🇸 السوق الأمريكي", "US", us_symbols)
    with box_sa:
        # ملاحظة: في ملف السعودية نخليها أرقام فقط أو .SR — نحن ننسق عند التحليل
        market_box("🇸🇦 السوق السعودي", "SA", sa_symbols)

with right:
    st.subheader("📊 التحليل")

    sel_market = st.session_state.get("selected_market")
    sel_symbol_raw = st.session_state.get("selected_symbol")

    if not sel_symbol_raw:
        st.info("اضغط على أي سهم من قائمة السوق (يسار) عشان يطلع التحليل هنا.")
    else:
        symbol = fmt_symbol(sel_symbol_raw, sel_market or "US")
        st.markdown(f"#### السهم المختار: *{symbol}*")

        with st.spinner("جاري جلب البيانات..."):
            df = fetch_history(symbol, PERIOD_OPTIONS[period])

        if df is None or df.empty:
            st.error("❌ ما لقيت بيانات لهذا السهم (ممكن الرمز خطأ أو ما فيه بيانات في Yahoo).")
        else:
            plan = calc_plan(df, risk_mode=risk_mode)
            if not plan:
                st.error("❌ البيانات غير كافية للتحليل (جرّب مدة أطول).")
            else:
                # Summary cards
                a, b, c, d = st.columns(4)
                a.metric("السعر الحالي", f"{plan['last']}")
                b.metric("الاتجاه", plan["trend"])
                c.metric("RSI", f"{plan['rsi']}")
                d.metric("ATR", f"{plan['atr']}")

                st.divider()

                ok = plan["suitable"]
                if ok:
                    st.success("✅ مناسب مبدئيًا للدخول حسب القواعد الحالية.")
                else:
                    st.warning("⚠️ غير مناسب مبدئيًا حسب القواعد الحالية (أو يحتاج تأكيد).")

                st.markdown("*الأسباب:*")
                for r in plan["reasons"]:
                    st.write("•", r)

                st.divider()

                st.markdown("### 🎯 خطة (مقترحة)")
                p1, p2, p3 = st.columns(3)
                p1.metric("سعر دخول مقترح", f"{plan['entry']}")
                p2.metric("وقف خسارة", f"{plan['stop']}")
                p3.metric("المخاطرة", f"{risk_mode} (ATR×{plan['atr_mult']}, قاع {plan['lookback']} يوم)")

                t1, t2, t3 = plan["targets"]
                st.markdown("*الأهداف:*")
                st.write(f"🎯 هدف 1: *{t1}*")
                st.write(f"🎯 هدف 2: *{t2}*")
                st.write(f"🎯 هدف 3: *{t3}*")

                st.divider()

                # Show last rows quick
                st.markdown("### 📌 آخر أسعار (مختصر)")
                show = df.tail(8).copy()
                show.index = show.index.astype(str)
                st.dataframe(show[["Open","High","Low","Close","Volume"]], use_container_width=True)

