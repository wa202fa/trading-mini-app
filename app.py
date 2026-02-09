import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# =========================
# Page
# =========================
st.set_page_config(page_title="Trading App", layout="wide")

# Hide sidebar بالكامل
st.markdown("""
<style>
[data-testid="stSidebar"] {display: none;}
[data-testid="collapsedControl"] {display: none;}
</style>
""", unsafe_allow_html=True)

# =========================
# Defaults (عدلها لاحقًا)
# =========================
US_DEFAULT = ["AAPL","NVDA","TSLA","MSFT","AMZN","GOOGL","META","NFLX","AMD","INTC","PLTR","AVGO","TSM","SPY","QQQ"]
SA_DEFAULT = ["1010.SR","1020.SR","1120.SR","1180.SR","2010.SR","2020.SR","2030.SR","2222.SR","2290.SR","2300.SR","2380.SR","4260.SR"]

# =========================
# Indicators
# =========================
def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

# =========================
# Data
# =========================
@st.cache_data(ttl=600)
def download_symbol(symbol: str, period: str):
    df = yf.download(symbol, period=period, progress=False, auto_adjust=False, threads=True)
    if df is None or df.empty:
        return None

    # Normalize yfinance MultiIndex columns if it happens
    if isinstance(df.columns, pd.MultiIndex):
        lvl1 = df.columns.get_level_values(1)
        if symbol in set(lvl1):
            df = df.xs(symbol, axis=1, level=1, drop_level=True)
        else:
            first_ticker = list(dict.fromkeys(lvl1))[0]
            df = df.xs(first_ticker, axis=1, level=1, drop_level=True)

    df = df.dropna()
    return None if df.empty else df

def compute_trend(last_close, ma20, ma50):
    if last_close > ma20 and last_close > ma50:
        return "صاعد"
    if last_close < ma20 and last_close < ma50:
        return "هابط"
    return "متذبذب"

def score_stock(last_close, ma20, ma50, rsi, vol, vol_ma20, mode: str):
    score = 0.0

    # Trend (40)
    if last_close > ma20 and ma20 > ma50:
        score += 40
    elif last_close > ma20:
        score += 25
    elif last_close > ma50:
        score += 15
    else:
        score += 5

    # RSI (30)
    if mode == "Swing":
        if 45 <= rsi <= 65:
            score += 30
        elif 35 <= rsi < 45 or 65 < rsi <= 72:
            score += 18
        elif rsi > 72:
            score += 6
        else:
            score += 10
    else:  # DayTrade
        if 35 <= rsi <= 60:
            score += 30
        elif 25 <= rsi < 35 or 60 < rsi <= 70:
            score += 18
        elif rsi > 70:
            score += 8
        else:
            score += 12

    # Volume (20)
    if np.isnan(vol_ma20) or vol_ma20 == 0:
        score += 8
    else:
        ratio = vol / vol_ma20
        if ratio >= 1.5:
            score += 20
        elif ratio >= 1.2:
            score += 16
        elif ratio >= 1.0:
            score += 12
        elif ratio >= 0.8:
            score += 8
        else:
            score += 4

    # Setup boost (10)
    if last_close > ma20 and rsi < 70:
        score += 10
    elif rsi < 35:
        score += 6
    else:
        score += 3

    return max(0, min(100, round(score, 1)))

def recommendation_from_score(score: float, trend: str, rsi: float):
    if score >= 75 and trend == "صاعد" and rsi < 72:
        return "BUY ✅"
    if score >= 60:
        return "WATCH 👀"
    return "WAIT ⏳"

def build_trade_plan(df: pd.DataFrame, mode: str):
    last_close = safe_float(df["Close"].iloc[-1])
    atr = calc_atr(df, 14)
    last_atr = safe_float(atr.iloc[-1])

    stop_mult = 1.0 if mode == "DayTrade" else 1.8
    if np.isnan(last_atr) or last_atr == 0:
        stop_pct = 0.03 if mode == "DayTrade" else 0.06
        stop = last_close * (1 - stop_pct)
    else:
        stop = last_close - (stop_mult * last_atr)

    risk = last_close - stop
    if risk <= 0:
        return None

    return {
        "Entry": round(last_close, 2),
        "Stop": round(stop, 2),
        "Risk (R)": round(risk, 2),
        "Target 1 (1R)": round(last_close + 1 * risk, 2),
        "Target 2 (2R)": round(last_close + 2 * risk, 2),
        "Target 3 (3R)": round(last_close + 3 * risk, 2),
    }

def analyze_symbol(symbol: str, period: str, mode: str):
    df = download_symbol(symbol, period)
    if df is None:
        return None

    close = df["Close"]
    vol = df["Volume"]

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    rsi = calc_rsi(close, 14)
    vol_ma20 = vol.rolling(20).mean()

    last_close = safe_float(close.iloc[-1])
    last_ma20 = safe_float(ma20.iloc[-1])
    last_ma50 = safe_float(ma50.iloc[-1])
    last_rsi = safe_float(rsi.iloc[-1])
    last_vol = safe_float(vol.iloc[-1])
    last_vol_ma20 = safe_float(vol_ma20.iloc[-1])

    trend = compute_trend(last_close, last_ma20, last_ma50)
    score = score_stock(last_close, last_ma20, last_ma50, last_rsi, last_vol, last_vol_ma20, mode)
    rec = recommendation_from_score(score, trend, last_rsi)
    plan = build_trade_plan(df, mode)

    out = {
        "Symbol": symbol,
        "Score": score,
        "Recommendation": rec,
        "Trend": trend,
        "Last Close": round(last_close, 2),
        "MA20": round(last_ma20, 2),
        "MA50": round(last_ma50, 2),
        "RSI": round(last_rsi, 2),
        "Vol/Avg20": round((last_vol / last_vol_ma20), 2) if (not np.isnan(last_vol_ma20) and last_vol_ma20 != 0) else np.nan,
    }
    if plan:
        out.update(plan)

    return out, df, plan

# =========================
# UI State
# =========================
if "market" not in st.session_state:
    st.session_state.market = None
if "chosen_symbol" not in st.session_state:
    st.session_state.chosen_symbol = None

# =========================
# UI Header
# =========================
st.title("Trading App")
st.subheader("الأسواق")

c1, c2 = st.columns(2)
with c1:
    if st.button("🇺🇸 السوق الأمريكي", use_container_width=True, key="market_us"):
        st.session_state.market = "US"
with c2:
    if st.button("🇸🇦 السوق السعودي", use_container_width=True, key="market_sa"):
        st.session_state.market = "SA"

st.markdown("---")

# =========================
# Market list (Auto analysis on selection)
# =========================
if st.session_state.market is None:
    st.info("اختر سوق عشان تظهر قائمة الأسهم.")
else:
    period = st.selectbox("المدة", ["3mo", "6mo", "1y", "2y"], index=1)
    mode = st.selectbox("نوع التحليل", ["DayTrade", "Swing"], index=1,
                        format_func=lambda x: "مضاربة" if x == "DayTrade" else "سوينق")

    if st.session_state.market == "US":
        st.markdown("### 🇺🇸 السوق الأمريكي")
        pick = st.radio("اختر سهم", US_DEFAULT, key="pick_us", label_visibility="collapsed")
    else:
        st.markdown("### 🇸🇦 السوق السعودي")
        pick = st.radio("اختر سهم", SA_DEFAULT, key="pick_sa", label_visibility="collapsed")

    # ✅ هذا هو الربط: مجرد الاختيار = يتحدث السهم + تحليل تلقائي
    st.session_state.chosen_symbol = pick

    st.markdown("---")
    st.success(f"✅ السهم المختار الآن: {st.session_state.chosen_symbol}")

    res = analyze_symbol(st.session_state.chosen_symbol, period=period, mode=mode)
    if not res:
        st.error("ما قدرت أجيب بيانات. تأكد من الرمز أو جرّب مدة ثانية.")
    else:
        out, df, plan = res

        st.subheader(f"{out['Symbol']} — {out['Recommendation']} (Score: {out['Score']})")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trend", out["Trend"])
        m2.metric("RSI", out["RSI"])
        m3.metric("Close", out["Last Close"])
        m4.metric("Vol/Avg20", out["Vol/Avg20"])

        st.markdown("### 🎯 دخول / وقف / أهداف")
        if plan:
            p1, p2, p3, p4, p5, p6 = st.columns(6)
            p1.metric("Entry", out["Entry"])
            p2.metric("Stop", out["Stop"])
            p3.metric("1R", out["Target 1 (1R)"])
            p4.metric("2R", out["Target 2 (2R)"])
            p5.metric("3R", out["Target 3 (3R)"])
            p6.metric("Risk", out["Risk (R)"])
        else:
            st.warning("ما قدرت أبني خطة (بيانات ATR غير كافية).")

        st.markdown("### 📊 الشارت")
        st.line_chart(df["Close"])
