import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="Trading App", page_icon="📈", layout="wide")

# استخدم مجلد التشغيل الحالي بدل _file_
BASE = Path.cwd()
US_PATH = BASE / "data" / "universe" / "us_symbols.txt"
SA_PATH = BASE / "data" / "universe" / "sa_symbols.txt"

DEFAULT_PERIOD = "6mo"
DEFAULT_INTERVAL = "1d"

# -----------------------
# Helpers
# -----------------------
def load_symbols(p: Path) -> list[str]:
    if not p.exists():
        return []
    syms = [l.strip() for l in p.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
    seen = set()
    out = []
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def calc_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill").fillna(50)

def calc_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr.fillna(method="bfill")

@st.cache_data(show_spinner=False, ttl=60*15)
def fetch_history(symbol: str, period: str = DEFAULT_PERIOD, interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
    t = yf.Ticker(symbol)
    df = t.history(period=period, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df

def trend_label(df: pd.DataFrame) -> str:
    if df.empty or len(df) < 30:
        return "غير واضح"
    sma20 = df["Close"].rolling(20).mean()
    sma50 = df["Close"].rolling(50).mean()
    last20 = sma20.iloc[-1]
    last50 = sma50.iloc[-1]
    if np.isnan(last20) or np.isnan(last50):
        return "غير واضح"
    if last20 > last50:
        return "صاعد"
    if last20 < last50:
        return "هابط"
    return "عرضي"

def fmt_symbol(sym: str, market: str) -> str:
    sym = sym.strip().upper()
    if market == "SA":
        if sym.endswith(".SR"):
            return sym
        if sym.isdigit():
            return f"{sym}.SR"
    return sym

def ensure_state():
    if "active_market" not in st.session_state:
        st.session_state.active_market = None
    if "chosen_symbol" not in st.session_state:
        st.session_state.chosen_symbol = None

ensure_state()

US_SYMBOLS = load_symbols(US_PATH)
SA_SYMBOLS = load_symbols(SA_PATH)

# -----------------------
# UI
# -----------------------
st.title("Trading App")
st.subheader("اختر السوق")

c1, c2 = st.columns(2)
with c1:
    if st.button("🇺🇸 السوق الأمريكي", use_container_width=True):
        st.session_state.active_market = "US"
        st.session_state.chosen_symbol = None
        st.rerun()

with c2:
    if st.button("🇸🇦 السوق السعودي", use_container_width=True):
        st.session_state.active_market = "SA"
        st.session_state.chosen_symbol = None
        st.rerun()

st.divider()

active = st.session_state.active_market

if active is None:
    st.info("اختر سوق من الأعلى لعرض قائمة الأسهم.")
    st.stop()

if active == "US":
    st.markdown("## 🇺🇸 السوق الأمريكي")
    if len(US_SYMBOLS) < 100:
        st.warning(f"قائمة أمريكا أقل من 100 سهم (الموجود: {len(US_SYMBOLS)})")
    options = US_SYMBOLS if US_SYMBOLS else ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
else:
    st.markdown("## 🇸🇦 السوق السعودي")
    if len(SA_SYMBOLS) < 100:
        st.warning(f"قائمة السعودية أقل من 100 سهم (الموجود: {len(SA_SYMBOLS)})")
    options = SA_SYMBOLS if SA_SYMBOLS else ["1010.SR", "1020.SR", "2010.SR", "2020.SR", "2030.SR"]

picked = st.selectbox("اختر السهم", options=options, index=0 if options else None)

symbol = fmt_symbol(picked, active)
st.session_state.chosen_symbol = symbol

st.success(f"✅ السهم المختار: {symbol}")
st.caption("يتم الآن جلب البيانات وتحليل السهم تلقائياً…")

with st.spinner("جاري التحليل..."):
    df = fetch_history(symbol)

if df.empty:
    st.error("❌ ما قدرنا نجلب بيانات لهذا السهم.")
    st.stop()

df["RSI"] = calc_rsi(df["Close"], 14)
df["ATR"] = calc_atr(df, 14)

price = float(df["Close"].iloc[-1])
rsi = float(df["RSI"].iloc[-1])
atr = float(df["ATR"].iloc[-1])
trend = trend_label(df)

entry = price
stop = price - (2.0 * atr)
r1 = price + (2.0 * atr)
r2 = price + (3.0 * atr)
r3 = price + (4.0 * atr)

msg = "مناسب مبدئياً للدخول."
if rsi >= 70:
    msg = "⚠️ RSI مرتفع (تشبع شراء)."
elif rsi <= 30:
    msg = "✅ RSI منخفض (تشبع بيع)."

st.divider()
st.markdown("## 📊 التحليل")

m1, m2, m3, m4 = st.columns(4)
m1.metric("السعر الحالي", f"{price:.2f}")
m2.metric("الاتجاه", trend)
m3.metric("RSI", f"{rsi:.1f}")
m4.metric("ATR", f"{atr:.2f}")

st.success(msg)

st.markdown("## 🎯 خطة الدخول")
p1, p2, p3, p4, p5 = st.columns(5)
p1.metric("دخول", f"{entry:.2f}")
p2.metric("وقف", f"{stop:.2f}")
p3.metric("هدف 1", f"{r1:.2f}")
p4.metric("هدف 2", f"{r2:.2f}")
p5.metric("هدف 3", f"{r3:.2f}")

with st.expander("عرض آخر البيانات"):
    st.dataframe(df.tail(30), use_container_width=True)
