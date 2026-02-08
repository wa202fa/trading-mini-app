import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================
# إعدادات عامة
# =========================
st.set_page_config(page_title="Trading Mini App", layout="wide")

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
# أدوات تحليل
# =========================
def to_tadawul_symbol(sym: str) -> str:
    s = sym.strip().upper()
    if s.endswith(".SR"):
        return s
    # إذا أرقام فقط (مثل 2222) نخليها 2222.SR
    if s.replace(".", "").isdigit():
        return f"{s}.SR"
    return s

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    series = series.astype(float)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low  = df["Low"].astype(float)
    close = df["Close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1/period, adjust=False).mean()

def safe_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None

# --- RSI Alerts helper ---
def rsi_alert_label(rsi_value, low=30, high=70):
    """
    يرجع: (emoji, label)
    """
    r = safe_float(rsi_value)
    if r is None:
        return "—", "غير متاح"
    if r <= low:
        return "🟢", "تشبع بيع"
    if r >= high:
        return "🟠", "تشبع شراء"
    return "🔵", "طبيعي"
# --- End RSI Alerts helper ---

def analyze_symbol(symbol: str, period: str):
    """
    يرجع (df, info) أو (None, None) لو فشل
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, auto_adjust=False)

        if df is None or df.empty:
            return None, None

        # تنظيف
        df = df.dropna(subset=["Close"]).copy()
        if df.empty:
            return None, None

        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()
        df["RSI14"] = rsi(df["Close"], 14)
        df["ATR14"] = atr(df, 14)

        last_close = safe_float(df["Close"].iloc[-1])
        last_rsi   = safe_float(df["RSI14"].iloc[-1])
        last_ma20  = safe_float(df["MA20"].iloc[-1])
        last_ma50  = safe_float(df["MA50"].iloc[-1])

        if last_ma20 is None or last_ma50 is None:
            trend = "—"
        else:
            trend = "صاعد" if last_ma20 > last_ma50 else ("هابط" if last_ma20 < last_ma50 else "محايد")

        info = {
            "Close": last_close,
            "RSI14": last_rsi,
            "MA20": last_ma20,
            "MA50": last_ma50,
            "Trend": trend,
        }
        return df, info
    except Exception:
        return None, None

def detect_entry_opportunity(df: pd.DataFrame, risk_level: str):
    """
    فرصة دخول بسيطة:
    - اتجاه صاعد (MA20 > MA50)
    - RSI بين 45 و 70
    - السعر فوق MA20
    يعطي: ok, reason, stop_price, score
    """
    p = RISK_PRESETS[risk_level]
    atr_mult = p["atr_mult"]

    close = df["Close"].astype(float)
    ma20 = df["MA20"].astype(float)
    ma50 = df["MA50"].astype(float)
    r = df["RSI14"].astype(float)
    a = df["ATR14"].astype(float)

    last_close = safe_float(close.iloc[-1])
    last_ma20  = safe_float(ma20.iloc[-1])
    last_ma50  = safe_float(ma50.iloc[-1])
    last_rsi   = safe_float(r.iloc[-1])
    last_atr   = safe_float(a.iloc[-1])

    if last_close is None or last_ma20 is None or last_ma50 is None or last_rsi is None:
        return False, "بيانات غير كافية", None, 0

    score = 0
    reasons = []

    uptrend = last_ma20 > last_ma50
    if uptrend:
        score += 2
    else:
        reasons.append("الاتجاه مو صاعد")

    if 45 <= last_rsi <= 70:
        score += 2
    else:
        reasons.append("RSI خارج النطاق (45-70)")

    if last_close >= last_ma20:
        score += 1
    else:
        reasons.append("السعر تحت MA20")

    # وقف خسارة: ATR * multiplier تحت السعر
    stop_price = None
    if last_atr is not None:
        stop_price = max(0.0, last_close - (last_atr * atr_mult))

    ok = score >= 4  # لازم 4/5
    reason = "، ".join(reasons) if reasons else "مطابق للشروط"
    return ok, reason, stop_price, score

def detect_breakout(df: pd.DataFrame, lookback: int, risk_level: str):
    """
    Breakout بسيط:
    - إغلاق اليوم > أعلى قمة خلال lookback يوم (باستثناء اليوم)
    - حجم اليوم >= متوسط حجم 20 يوم (اختياري إن توفر)
    وقف الخسارة: تحت مستوى الكسر أو ATR
    """
    p = RISK_PRESETS[risk_level]
    atr_mult = p["atr_mult"]

    if len(df) < max(lookback + 2, 25):
        return False, "بيانات غير كافية", None, None

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    vol = df["Volume"] if "Volume" in df.columns else None
    a = df["ATR14"].astype(float) if "ATR14" in df.columns else None

    last_close = safe_float(close.iloc[-1])
    prev_highs = high.iloc[-(lookback+1):-1]  # آخر lookback بدون اليوم
    level = safe_float(prev_highs.max())

    if last_close is None or level is None:
        return False, "بيانات غير كافية", None, None

    vol_ok = True
    if vol is not None:
        v = pd.to_numeric(vol, errors="coerce")
        last_v = safe_float(v.iloc[-1])
        v_avg = safe_float(v.rolling(20).mean().iloc[-1])
        if (last_v is not None) and (v_avg is not None) and (v_avg > 0):
            vol_ok = last_v >= v_avg * 0.9  # تساهل بسيط

    is_break = (last_close > level) and vol_ok

    b_stop = None
    last_atr = safe_float(a.iloc[-1]) if a is not None else None
    if last_atr is not None:
        b_stop = max(0.0, last_close - (last_atr * atr_mult))
    if level is not None and b_stop is not None:
        b_stop = min(b_stop, level)  # خلي الوقف على الأقل تحت مستوى الكسر

    reason = "كسر مقاومة + حجم جيد" if is_break else "ما تحقق الكسر/الحجم"
    return is_break, reason, level, b_stop

# =========================
# واجهة التطبيق
# =========================
st.title("📈 Trading Mini App (US + Saudi Tadawul)")
st.caption("تحليل بسيط + RSI + اتجاه + قائمة متابعة + تنبيه فرصة دخول + Breakout")

# Sidebar
with st.sidebar:
    st.header("⚙️ الإعدادات")

    market = st.selectbox("السوق", ["أمريكي", "سعودي"], index=0)
    symbol_input = st.text_input("اكتب الرمز", value="AAPL" if market == "أمريكي" else "2222")
    period = st.selectbox("اختر المدة", list(PERIOD_OPTIONS.keys()), index=2)  # 6mo
    risk_level = st.selectbox("وضع المخاطرة", list(RISK_PRESETS.keys()), index=1)  # متوسط

    enable_entry = st.toggle("تفعيل تنبيه فرصة الدخول", value=True)
    enable_breakout = st.toggle("تفعيل تنبيه كسر مقاومة (Breakout)", value=True)
    breakout_lookback = st.selectbox("فترة الكسر", [10, 20, 30, 50, 60], index=1)

    st.markdown("---")
    st.subheader("📋 قائمة المتابعة")
    watchlist_text = st.text_area("القائمة (سطر لكل سهم)", value="AAPL\nNVDA\n2222", height=110)
    scan_btn = st.button("🚀 افحص القائمة", use_container_width=True)

# تجهيز الرمز حسب السوق
symbol = symbol_input.strip().upper()
if market == "سعودي":
    symbol = to_tadawul_symbol(symbol)

# =========================
# تحليل السهم (فردي)
# =========================
st.markdown("---")
st.markdown("## 📊 تحليل السهم")

colA, colB = st.columns([1, 4])
with colA:
    run_single = st.button("🔍 حلّل السهم", use_container_width=True)

if run_single:
    st.write(f"جاري جلب البيانات لـ: {symbol}")
    df, info = analyze_symbol(symbol, PERIOD_OPTIONS[period])

    if df is None or info is None:
        st.error("❌ ما قدرت أجيب بيانات السهم. (تأكدي من الرمز + الإنترنت)")
    else:
        st.success("✅ تم جلب البيانات بنجاح")

        # --- Chart (analysis button) ---
        try:
            _df = df.copy()
            cols = {c.lower(): c for c in _df.columns}
            def pick(key):
                for k,v in cols.items():
                    if k == key or key in k:
                        return v
                return None
            o = pick("open"); h = pick("high"); l = pick("low"); c = pick("close")
            if all([o,h,l,c]) and len(_df) > 5:
                fig = go.Figure(data=[go.Candlestick(
                    x=_df.index,
                    open=_df[o], high=_df[h], low=_df[l], close=_df[c],
                )])
                fig.update_layout(height=420, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=30,b=10))
                st.subheader("📈 الشارت")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ما قدرت أرسم الشارت لأن البيانات ما فيها أعمدة Open/High/Low/Close بشكل واضح.")
        except Exception as e:
            st.warning(f"تعذر رسم الشارت: {e}")
        # --- End Chart ---


        # مؤشرات أعلى الصفحة
        c1, c2, c3 = st.columns(3)
        c1.metric("السعر الحالي", f"{info['Close']:.2f}" if info["Close"] is not None else "—")
        c2.metric("RSI", f"{info['RSI14']:.2f}" if info["RSI14"] is not None else "—")
        c3.metric("الاتجاه", info["Trend"])

        # ✅ تنبيه RSI (واضح ومضمون)
        st.subheader("🔔 تنبيه RSI")
        r = info.get("RSI14", None)
        emoji, label = rsi_alert_label(r)
        if label == "تشبع بيع":
            st.success(f"{emoji} {label} — ممكن ارتداد (RSI منخفض)")
        elif label == "تشبع شراء":
            st.warning(f"{emoji} {label} — انتبه من تصحيح (RSI مرتفع)")
        elif label == "طبيعي":
            st.info(f"{emoji} {label} — ما فيه تشبع واضح")
        else:
            st.info("🔔 RSI غير متاح")

        # إشارات
        st.subheader("الإشارة")
        if enable_entry:
            ok, reason, stop_p, score = detect_entry_opportunity(df, risk_level)
            if ok:
                st.success(f"✅ تنبيه فرصة الدخول — الدرجة: {score}/5 — وقف خسارة تقريبي: {stop_p:.2f}" if stop_p else f"✅ تنبيه فرصة الدخول — الدرجة: {score}/5")
            else:
                st.info(f"ℹ️ فرصة الدخول غير متحققة — ({reason})")

        if enable_breakout:
            b_ok, b_reason, level, b_stop = detect_breakout(df, int(breakout_lookback), risk_level)
            if b_ok:
                msg = f"🚀 Breakout — مستوى الكسر: {level:.2f}"
                if b_stop is not None:
                    msg += f" — وقف خسارة: {b_stop:.2f}"
                st.success(msg)
            else:
                st.info("ℹ️ Breakout غير متحقق حالياً")

        st.markdown("### جدول الأسعار")
        # عرض مرتب: آخر 60 صف فقط
        show = df[["Open", "High", "Low", "Close", "Volume"]].tail(60).copy()
        show.index = pd.to_datetime(show.index).date
        st.dataframe(show, use_container_width=True)

# =========================
# فحص قائمة المتابعة
# =========================
st.markdown("---")
st.markdown("## 🔎 فحص قائمة المتابعة")

def parse_watchlist(txt: str):
    items = [x.strip() for x in (txt or "").splitlines() if x.strip()]
    # حذف التكرار مع الحفاظ على الترتيب
    seen = set()
    out = []
    for it in items:
        u = it.upper()
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

if scan_btn:
    items = parse_watchlist(watchlist_text)
    if not items:
        st.error("اكتبي أسهم في القائمة أولاً.")
        st.stop()

    rows = []
    for raw in items:
        sym = raw.strip().upper()
        if market == "سعودي":
            sym = to_tadawul_symbol(sym)

        df, info = analyze_symbol(sym, PERIOD_OPTIONS[period])
        if df is None or info is None:
            rows.append({
                "الرمز": sym,
                "الاتجاه": "—",
                "RSI": "—",
                "تنبيه RSI": "—",
                "فرصة دخول؟": "❌",
                "Breakout؟": "—",
                "وقف خسارة": "—",
            })
            continue

        # Entry + Breakout
        ok, reason, stop_p, score = detect_entry_opportunity(df, risk_level) if enable_entry else (False, "", None, 0)
        b_ok, b_reason, level, b_stop = detect_breakout(df, int(breakout_lookback), risk_level) if enable_breakout else (False, "", None, None)
        final_stop = b_stop if b_ok else stop_p

        # RSI Alert for table
        emoji, label = rsi_alert_label(info.get("RSI14", None))
        rsi_text = f"{emoji} {label}" if label != "غير متاح" else "—"

        rows.append({
            "الرمز": sym,
            "الاتجاه": info["Trend"],
            "RSI": f"{info['RSI14']:.2f}" if info["RSI14"] is not None else "—",
            "تنبيه RSI": rsi_text,
            "فرصة دخول؟": "✅" if ok else "❌",
            "Breakout؟": "🚀" if b_ok else "—",
            "وقف خسارة": f"{final_stop:.2f}" if final_stop is not None else "—",
        })

    out = pd.DataFrame(rows)

    # ترتيب أعمدة ثابت
    preferred = ["الرمز", "الاتجاه", "RSI", "تنبيه RSI", "فرصة دخول؟", "Breakout؟", "وقف خسارة"]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    out = out[cols]

    st.dataframe(out, use_container_width=True)

    st.markdown("### ⭐ الفرص فقط")
    entry_col = "فرصة دخول؟"
    brk_col = "Breakout؟"

    mask = pd.Series([False] * len(out), index=out.index)
    if enable_entry and entry_col in out.columns:
        mask = mask | out[entry_col].astype(str).str.contains("✅")
    if enable_breakout and brk_col in out.columns:
        mask = mask | out[brk_col].astype(str).str.contains("🚀")

    only_ok = out[mask]
    if only_ok.empty:
        st.info
