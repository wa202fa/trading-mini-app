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
    # لو كتب .SR خلاص لا نضيفها
    if s.endswith(".SR"):
        return s
    # تداول غالبًا أرقام
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
    """
    يصلّح اختلافات yfinance:
    - DataFrame فاضي
    - أعمدة MultiIndex مثل ('Close','2222.SR')
    - close / adj close بحروف مختلفة
    """
    if df is None or df.empty:
        return None, f"ما قدرت أجيب بيانات للرمز: {symbol} (يمكن الرمز غلط أو مافي بيانات بالفترة)."

    # لو الأعمدة MultiIndex (مثلاً ('Close','2222.SR')) نخليها أسماء بسيطة
    if hasattr(df.columns, "levels") and len(getattr(df.columns, "levels", [])) > 1:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # تنظيف أسماء الأعمدة
    df.columns = [str(c).strip() for c in df.columns]

    # إذا ما فيه Close جرّب بدائل
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
    # تحميل البيانات
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)

    # توحيد/تصليح الأعمدة
    df, err = normalize_yf_df(df, symbol)
    if err:
        return None, err

    # حساب المؤشرات
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI14"] = compute_rsi(df["Close"], 14)

    last_close = float(df["Close"].iloc[-1])
    ma20 = float(df["MA20"].iloc[-1]) if not pd.isna(df["MA20"].iloc[-1]) else None
    ma50 = float(df["MA50"].iloc[-1]) if not pd.isna(df["MA50"].iloc[-1]) else None
    rsi = float(df["RSI14"].iloc[-1]) if not pd.isna(df["RSI14"].iloc[-1]) else None

    # اتجاه بسيط
    trend = "محايد"
    if ma20 is not None and ma50 is not None:
        if ma20 > ma50:
            trend = "الاتجاه: صاعد"
        elif ma20 < ma50:
            trend = "الاتجاه: هابط"

    # حالة RSI
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

    run_btn = st.button("حلّل السهم الآن", type="primary")


with col_right:
    st.markdown("### ملاحظات سريعة")
    st.write("• السوق السعودي: اكتب الرقم فقط مثل 1120 أو 2222.")
    st.write("• التطبيق يحوّل تلقائيًا إلى ‎.SR.")
    st.write("• البيانات عبر Yahoo Finance باستخدام yfinance.")


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