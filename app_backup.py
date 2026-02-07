import streamlit as st
import yfinance as yf
import pandas as pd
from io import BytesIO


# =========================
# إعدادات الصفحة
# =========================
st.set_page_config(page_title="Trading Mini App", layout="wide")
st.title("📈 Trading Mini App (US + Saudi Tadawul)")


# =========================
# إعدادات المدد
# =========================
PERIOD_OPTIONS = {
    "1mo": "1mo",
    "3mo": "3mo",
    "6mo": "6mo",
    "1y": "1y",
    "2y": "2y",
    "5y": "5y",
    "max": "max",
}


# =========================
# أدوات مساعدة
# =========================
def to_tadawul_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if not s:
        return s
    if s.endswith(".SR"):
        return s
    if s.isdigit():
        return f"{s}.SR"
    return s


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=True, sheet_name="data")
    return output.getvalue()


def normalize_yf_df(df: pd.DataFrame, symbol: str):
    if df is None or df.empty:
        return None, f"ما قدرت أجيب بيانات للرمز: {symbol} (يمكن الرمز غلط أو مافي بيانات بالفترة)."

    if hasattr(df.columns, "levels") and len(getattr(df.columns, "levels", [])) > 1:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df.columns = [str(c).strip() for c in df.columns]

    if "Close" not in df.columns:
        lower_map = {c.lower(): c for c in df.columns}
        if "adj close" in lower_map:
            df["Close"] = df[lower_map["adj close"]]
        elif "close" in lower_map:
            df["Close"] = df[lower_map["close"]]
        else:
            return None, f"البيانات رجعت بدون Close للرمز: {symbol}. الأعمدة: {list(df.columns)[:10]}"

    return df, None


def analyze_symbol(symbol: str, period: str):
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)

    df, err = normalize_yf_df(df, symbol)
    if err:
        return None, err

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI14"] = compute_rsi(df["Close"], 14)

    last_close = float(df["Close"].iloc[-1])
    ma20 = float(df["MA20"].iloc[-1]) if not pd.isna(df["MA20"].iloc[-1]) else None
    ma50 = float(df["MA50"].iloc[-1]) if not pd.isna(df["MA50"].iloc[-1]) else None
    rsi = float(df["RSI14"].iloc[-1]) if not pd.isna(df["RSI14"].iloc[-1]) else None

    trend = "محايد"
    if ma20 is not None and ma50 is not None:
        if ma20 > ma50:
            trend = "الاتجاه: صاعد"
        elif ma20 < ma50:
            trend = "الاتجاه: هابط"

    rsi_state = "غير متاح"
    if rsi is not None:
        if rsi >= 70:
            rsi_state = "تشبّع شرائي"
        elif rsi <= 30:
            rsi_state = "تشبّع بيعي"
        else:
            rsi_state = "طبيعي"

    summary = {
        "آخر إغلاق": last_close,
        "MA20": ma20,
        "MA50": ma50,
        "RSI14": rsi,
        "اتجاه": trend,
        "حالة RSI": rsi_state,
        "عدد الأيام": int(df.shape[0]),
    }

    return df, summary


def detect_entry_opportunity(df: pd.DataFrame):
    """
    تنبيه دخول بسيط (Daily):
    - Uptrend: MA20 > MA50
    - السعر فوق MA20
    - RSI بين 45 و 70 (مو متضخم ولا ضعيف)
    يرجّع (is_opportunity, reason, suggested_stop)
    """
    if df is None or df.empty:
        return False, "ما فيه بيانات", None

    if df.shape[0] < 55:
        return False, "البيانات قليلة للتحليل (نحتاج على الأقل ~55 يوم)", None

    last = df.iloc[-1]
    ma20 = last.get("MA20")
    ma50 = last.get("MA50")
    close = last.get("Close")
    rsi = last.get("RSI14")

    if pd.isna(ma20) or pd.isna(ma50) or pd.isna(close) or pd.isna(rsi):
        return False, "المؤشرات ما اكتملت (جرّب مدة أطول)", None

    uptrend = (ma20 > ma50)
    above_ma20 = (close > ma20)

    if not uptrend:
        return False, "مو صاعد (MA20 مو أعلى من MA50)", None
    if not above_ma20:
        return False, "السعر تحت MA20 (انتظر يرجع فوق)", None

    if rsi < 45:
        return False, "RSI منخفض (ضعف زخم)", None
    if rsi > 70:
        return False, "RSI مرتفع (تشبّع شرائي)", None

    # اقتراح وقف خسارة بسيط: 1.5% تحت MA20
    suggested_stop = float(ma20) * 0.985
    return True, "صاعد + زخم طبيعي (فرصة محتملة للدخول)", suggested_stop


# =========================
# واجهة المستخدم
# =========================
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.markdown("### اختر السوق")
    market = st.selectbox("السوق", ["أمريكي", "سعودي (تداول)"], index=1)

    st.markdown("### أدخل الرمز")
    st.caption("أمريكي: اكتب مثل AAPL, NVDA, MSFT — سعودي: اكتب مثل 1120 أو 2222 (التطبيق يحولها إلى .SR)")
    symbol_input = st.text_input("الرمز", value="2222")

    st.markdown("### المدة")
    period = st.selectbox("المدة", list(PERIOD_OPTIONS.keys()), index=2)

    st.markdown("### التنبيهات")
    alerts_on = st.toggle("فعّل تنبيه فرصة الدخول", value=True)
    st.caption("التنبيه يظهر داخل التطبيق عند التحليل إذا توفرت شروط دخول بسيطة.")

    run_btn = st.button("حلّل السهم الآن", type="primary")


with col_right:
    st.markdown("### ملاحظات سريعة")
    st.write("• السوق السعودي: اكتب الرقم فقط مثل 1120 أو 2222.")
    st.write("• التطبيق يحوّل تلقائيًا إلى ‎.SR.")
    st.write("• البيانات عبر Yahoo Finance باستخدام yfinance.")
    st.write("• التنبيه الحالي بسيط: MA20/MA50 + RSI + السعر فوق MA20.")


# =========================
# تنفيذ التحليل
# =========================
if run_btn:
    raw = symbol_input.strip()
    if not raw:
        st.error("اكتب رمز أولاً.")
        st.stop()

    symbol = raw.upper()
    if market.startswith("سعودي"):
        symbol = to_tadawul_symbol(symbol)

    with st.spinner("جاري التحليل..."):
        df, result = analyze_symbol(symbol, PERIOD_OPTIONS[period])

    if df is None:
        st.error(result)
        st.stop()

    # بطاقات سريعة
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("آخر إغلاق", f"{result['آخر إغلاق']:.2f}")
    c2.metric("RSI14", f"{result['RSI14']:.2f}" if result["RSI14"] is not None else "—")
    c3.metric("MA20", f"{result['MA20']:.2f}" if result["MA20"] is not None else "—")
    c4.metric("MA50", f"{result['MA50']:.2f}" if result["MA50"] is not None else "—")

    # شريط إشارة بسيط
    st.markdown("## 📊 الإشارة")
    msg = f"{result['اتجاه']} — RSI: {result['حالة RSI']} — عدد الأيام: {result['عدد الأيام']}"
    if "صاعد" in result["اتجاه"]:
        st.success(msg)
    elif "هابط" in result["اتجاه"]:
        st.warning(msg)
    else:
        st.info(msg)

    # =========================
    # تنبيه فرصة الدخول
    # =========================
    if alerts_on:
        is_ok, reason, stop_price = detect_entry_opportunity(df)
        st.markdown("## 🔔 تنبيه فرصة الدخول")
        if is_ok:
            st.success(f"✅ {reason}")
            st.write(f"*اقتراح وقف خسارة (تقريبي):* {stop_price:.2f}")
            st.caption("تنبيه تقني فقط وليس توصية شراء. استخدم إدارة مخاطر.")
        else:
            st.info(f"لا يوجد تنبيه دخول الآن: {reason}")

    # الرسم
    st.markdown("## 📈 الرسم البياني")
    chart_df = df[["Close", "MA20", "MA50"]].copy()
    st.line_chart(chart_df, width="stretch")

    # عرض البيانات + إكسل
    with st.expander("عرض البيانات (آخر 30 صف)"):
        st.dataframe(df.tail(30), width="stretch")

    excel_bytes = df_to_excel_bytes(df)
    st.download_button(
        "تحميل البيانات Excel",
        data=excel_bytes,
        file_name=f"{symbol}_{period}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
