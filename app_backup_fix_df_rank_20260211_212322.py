import streamlit as st






# === MOBILE_UI_START ===
import math

def _mobile_ui_css():
    st.markdown("""
<style>
/* Mobile-first container */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1100px; }

/* Softer typography */
h1,h2,h3 { letter-spacing: .3px; }
small, .muted { opacity:.75; }

/* Card system */
.card {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 14px 14px;
  margin: 10px 0;
  backdrop-filter: blur(10px);
}
.card-title { font-size: 16px; font-weight: 800; margin: 0 0 6px 0; }
.card-sub { font-size: 12px; opacity: .8; margin: 0 0 10px 0; }
.pill {
  display:inline-block; padding: 4px 10px; border-radius: 999px;
  background: rgba(34,197,94,.15); border: 1px solid rgba(34,197,94,.35);
  font-size: 12px; font-weight: 700; margin-left: 6px;
}
.pill-warn {
  background: rgba(245,158,11,.12); border: 1px solid rgba(245,158,11,.35);
}
.kv { display:flex; gap:10px; flex-wrap:wrap; }
.kv span{
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px;
  padding: 7px 10px;
  font-size: 12px;
}
.hr { border-top: 1px solid rgba(255,255,255,0.08); margin: 14px 0; }

/* Buttons */
.stButton>button { border-radius: 12px; font-weight: 800; padding: 0.55rem 0.9rem; }

/* Make tables less "article-like" on mobile */
[data-testid="stDataFrame"] { border-radius: 14px; overflow:hidden; border: 1px solid rgba(255,255,255,0.08); }

/* Responsive: stack on small widths */
@media (max-width: 780px) {
  .block-container { padding-left: 0.9rem; padding-right: 0.9rem; }
}
</style>
""", unsafe_allow_html=True)

def render_opportunity_cards(df, top_n=25, title="أفضل الفرص"):
    """
    يحول جدول الفرص إلى بطاقات Mobile-First.
    يحاول يلقط الأعمدة الشائعة (Symbol/Price/RSI/ATR/Score/Risk/Opportunity).
    """
    try:
        import pandas as pd
    except Exception:
        pd = None

    if df is None:
        st.info("ما فيه بيانات لعرضها.")
        return

    # إذا جتك ليست بدل DF
    try:
        shape = getattr(df, "shape", None)
    except Exception:
        shape = None

    st.markdown(f"### 🧾 {title}")
    st.caption("عرض سريع كبطاقات (مناسب للجوال). افتح التفاصيل لكل سهم إذا تبي.")

    # قص العدد
    try:
        df2 = df.head(top_n)
    except Exception:
        df2 = df

    # كشف أسماء أعمدة مرنة
    def pick(cols, candidates):
        cols_l = [c.lower() for c in cols]
        for cand in candidates:
            if cand in cols_l:
                return cols[cols_l.index(cand)]
        return None

    cols = list(getattr(df2, "columns", [])) if hasattr(df2, "columns") else []
    sym_col = pick(cols, ["symbol","ticker","رمز","السهم"])
    price_col = pick(cols, ["price","close","السعر"])
    score_col = pick(cols, ["score","scoring","rank_score","التقييم"])
    opp_col = pick(cols, ["opportunity","signal","الفرصة","فرصة"])
    risk_col = pick(cols, ["risk","risk %","risk%","risk_pct","المخاطرة"])
    rsi_col  = pick(cols, ["rsi"])
    atr_col  = pick(cols, ["atr"])
    ret_col  = pick(cols, ["return 20d %","return20d","ret_20d","return","العائد"])

    # تحويل لصفوف
    rows = []
    try:
        it = df2.iterrows()
        for _, r in it:
            rows.append(r)
    except Exception:
        # fallback: try list of dicts
        try:
            rows = list(df2)
        except Exception:
            rows = []

    if not rows:
        st.info("البيانات فاضية.")
        return

    for i, r in enumerate(rows, start=1):
        def getv(col, default="—"):
            if col is None:
                return default
            try:
                v = r[col]
            except Exception:
                try:
                    v = r.get(col, default)
                except Exception:
                    return default
            if v is None:
                return default
            # تنسيق أرقام
            try:
                if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                    # سعر و نسب
                    if col in [price_col]:
                        return f"{v:.2f}"
                    if col in [risk_col, ret_col]:
                        return f"{v:.2f}%"
                    if col in [rsi_col, atr_col, score_col]:
                        return f"{v:.2f}"
                return str(v)
            except Exception:
                return str(v)

        sym = getv(sym_col, f"#{i}")
        price = getv(price_col)
        score = getv(score_col)
        opp = getv(opp_col)
        risk = getv(risk_col)
        rsi = getv(rsi_col)
        atr = getv(atr_col)
        ret = getv(ret_col)

        # شارة بسيطة
        badge = "قوي"
        badge_cls = "pill"
        try:
            sc = float(score) if score not in ("—","") else None
            if sc is not None and sc >= 1.35:
                badge = "قوي جدًا"
                badge_cls = "pill"
            elif sc is not None and sc < 1.15:
                badge = "متوسط"
                badge_cls = "pill pill-warn"
        except Exception:
            pass

        st.markdown(f"""
<div class="card">
  <div class="card-title">{i}. {sym} <span class="{badge_cls}">{badge}</span></div>
  <div class="card-sub">سعر: <b>{price}</b> • سكور: <b>{score}</b> • مخاطرة: <b>{risk}</b></div>
  <div class="kv">
    <span>Opportunity: <b>{opp}</b></span>
    <span>RSI: <b>{rsi}</b></span>
    <span>ATR: <b>{atr}</b></span>
    <span>Return(20d): <b>{ret}</b></span>
  </div>
</div>
""", unsafe_allow_html=True)

def _mobile_ui_boot():
    # لازم تتنادى بعد set_page_config
    _mobile_ui_css()

# === MOBILE_UI_END ===


# === PREMIUM_HOME_UI_START ===
def _apply_premium_home_ui():
    import streamlit as st

    st.markdown("""
    <style>
      /* Global polish */
      .stApp { font-family: ui-sans-serif, system-ui, -apple-system, "SF Pro Display", "Segoe UI", Arial; }
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}

      /* Better spacing */
      .block-container { padding-top: 1.8rem; padding-bottom: 2.5rem; max-width: 1200px; }

      /* Cards look for vertical blocks */
      div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stHorizontalBlock"]) {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 16px 16px 10px 16px;
      }

      /* Inputs */
      div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.06) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
      }
      input, textarea {
        background: rgba(255,255,255,0.06) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
      }

      /* Buttons */
      .stButton > button {
        border-radius: 14px !important;
        font-weight: 700 !important;
        padding: 10px 14px !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
      }

      /* Headings */
      h1,h2,h3 { letter-spacing: .3px; }
      h1 { font-size: 38px !important; margin-bottom: 4px !important; }
      h2 { font-size: 24px !important; margin-top: 10px !important; }

      /* Small badges */
      .badge {
        display:inline-flex; align-items:center; gap:8px;
        padding:6px 10px; border-radius:999px;
        background: rgba(34,197,94,0.14);
        border: 1px solid rgba(34,197,94,0.28);
        font-size: 12px; font-weight: 700;
      }
      .badge.gray{
        background: rgba(148,163,184,0.10);
        border: 1px solid rgba(148,163,184,0.22);
      }

      /* Table polish */
      .stDataFrame, .stTable { border-radius: 16px; overflow: hidden; }
    </style>
    """, unsafe_allow_html=True)

    # Hero header
    st.markdown("""
    <div style="
      padding:18px 18px 16px 18px;
      border-radius:20px;
      border:1px solid rgba(255,255,255,0.10);
      background: radial-gradient(900px 420px at 15% 0%, rgba(34,197,94,0.16), transparent 55%),
                  radial-gradient(900px 420px at 85% 0%, rgba(59,130,246,0.16), transparent 55%),
                  rgba(255,255,255,0.03);
      ">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:14px; flex-wrap:wrap;">
        <div>
          <div style="font-size:34px; font-weight:800; line-height:1.15;">Trading App</div>
          <div style="opacity:.75; font-size:13px; margin-top:4px;">واجهة رئيسية — فرص اليوم + فلترة سريعة</div>
        </div>
        <div style="display:flex; gap:10px; align-items:center;">
          <span class="badge">⚡ Top Opportunities</span>
          <span class="badge gray">🧠 Smart Filters</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

_apply_premium_home_ui()
# === PREMIUM_HOME_UI_END ===


# === UI_CSS_START ===
st.markdown(r"""
<style>
/* تغيير واضح جدًا */
[data-testid="stAppViewContainer"]{
  background: radial-gradient(900px 500px at 20% 10%, rgba(34,197,94,.18), transparent 55%),
              radial-gradient(900px 500px at 90% 10%, rgba(59,130,246,.18), transparent 55%),
              linear-gradient(180deg,#070a12 0%, #0b1220 100%) !important;
}
.block-container { padding-top: 2rem !important; max-width: 1200px !important; }
h1,h2,h3 { letter-spacing:.4px !important; }

.card{
  background: rgba(255,255,255,.06) !important;
  border: 1px solid rgba(255,255,255,.10) !important;
  border-radius: 16px !important;
  padding: 16px !important;
}
.stButton > button{
  border-radius: 12px !important;
  font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)
# === UI_CSS_END ===

# === UI_CSS_START ===
st.markdown("""
<style>
.stApp {
  background: linear-gradient(180deg, #0e1117 0%, #0b1220 100%);
}
.block-container { padding-top: 2rem; }
h1, h2, h3 { letter-spacing: 1px; }
.card {
  background: rgba(255,255,255,0.05);
  border-radius: 14px;
  padding: 16px;
  margin-bottom: 12px;
  border: 1px solid rgba(255,255,255,0.08);
}
.stButton>button {
  border-radius: 10px;
  font-weight: 600;
}
</style>
""", unsafe_allow_html=True)
# === UI_CSS_END ===

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import logging, contextlib, io

# كتم رسائل yfinance المزعجة
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('requests').setLevel(logging.CRITICAL)

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="Trading App", page_icon="📈", layout="wide")

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
    seen, out = set(), []
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
    return rsi.bfill().fillna(50)

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
    return atr.bfill()

@st.cache_data(show_spinner=False, ttl=60*15)
def fetch_history(symbol: str, period: str = DEFAULT_PERIOD, interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
    try:
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            t = yf.Ticker(symbol)
            df = t.history(period=period, interval=interval, auto_adjust=False)
    except Exception:
        return pd.DataFrame()

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

# -----------------------
# State
# -----------------------
if "active_market" not in st.session_state:
    st.session_state.active_market = None

# الافتراضي: مضاربة
if "style" not in st.session_state:
    st.session_state.style = "مضاربة"

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
        st.rerun()
with c2:
    if st.button("🇸🇦 السوق السعودي", use_container_width=True):
        st.session_state.active_market = "SA"
        st.rerun()

st.divider()

active = st.session_state.active_market
if active is None:
    st.info("اختر سوق من الأعلى.")
    st.stop()

# اختيار الأسلوب (الافتراضي مضاربة)
style = st.selectbox("أسلوب التداول", ["مضاربة", "سوينق", "متوازن", "محافظ"], index=0)
st.session_state.style = style

if active == "US":
    options = US_SYMBOLS if US_SYMBOLS else ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
else:
    options = SA_SYMBOLS if SA_SYMBOLS else ["1010.SR", "1020.SR", "2010.SR", "2020.SR", "2030.SR"]

st.divider()
st.markdown("## 🏆 ترتيب الفرص")

if st.button("🔍 تحليل وترتيب جميع الأسهم"):
    rows = []
    with st.spinner("جاري التحليل..."):
        for sym in options:
            symbol = fmt_symbol(sym, active)
            df = fetch_history(symbol)
            if df.empty or len(df) < 60:
                continue

            df["RSI"] = calc_rsi(df["Close"])
            df["ATR"] = calc_atr(df)

            price = float(df["Close"].iloc[-1])
            rsi = float(df["RSI"].iloc[-1])
            atr = float(df["ATR"].iloc[-1])
            trend = trend_label(df)

            price_20 = float(df["Close"].iloc[-20])
            ret_20 = (price / price_20 - 1) * 100.0
            risk = (atr / price) * 100.0

            # Scoring حسب الأسلوب
            score = 0.0

            if style == "مضاربة":
                if 20 <= rsi <= 40:
                    score += 3
                if trend == "صاعد":
                    score += 2
                score += ret_20 / 15.0
                score -= risk

            elif style == "سوينق":
                if trend == "صاعد":
                    score += 3
                if 40 <= rsi <= 60:
                    score += 2
                score += ret_20 / 10.0
                score -= risk / 2

            elif style == "متوازن":
                if trend == "صاعد":
                    score += 2
                if 35 <= rsi <= 65:
                    score += 2
                score += ret_20 / 12.0
                score -= risk / 1.5

            else:  # محافظ
                if trend == "صاعد":
                    score += 2
                if risk < 3:
                    score += 2
                score += ret_20 / 20.0
                score -= risk

            rows.append({
                "Symbol": symbol,
                "Price": round(price, 2),
                "Trend": trend,
                "RSI": round(rsi, 1),
                "ATR": round(atr, 2),
                "Return 20d %": round(ret_20, 2),
                "Risk %": round(risk, 2),
                "Score": round(score, 2),
            })

    if not rows:
        st.warning("ما قدرنا نطلع نتائج.")
        st.stop()

    df_rank = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)

    # تصنيف الفرص
    q1 = df_rank["Score"].quantile(0.7)
    q0 = df_rank["Score"].quantile(0.4)

    def classify(s):
        if s >= q1:
            return "قوي"
        if s >= q0:
            return "متوسط"
        return "ضعيف"

    df_rank["Opportunity"] = df_rank["Score"].apply(classify)
    df_rank.insert(0, "Flag", df_rank["Opportunity"].map({
        "قوي": "🟩 قوي",
        "متوسط": "🟨 متوسط",
        "ضعيف": "⬜️ ضعيف"
    }))

    st.success("✅ تم ترتيب الأسهم حسب أفضل الفرص")
    st.caption("🟩 قوي = فرص أفضل حسب الأسلوب المختار")

# --- Mobile cards (auto) ---
try:
    _df_for_cards = df_rank
except Exception:
    _df_for_cards = None

render_opportunity_cards(_df_for_cards, top_n=25, title="ترتيب الفرص")

with st.expander("عرض الجدول الكامل (اختياري)"):
    st.dataframe(df_rank, use_container_width=True)

