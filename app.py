import streamlit as st
import yfinance as yf
import pandas as pd
from io import BytesIO

# --------------------------
# إعدادات الصفحة
# --------------------------
st.set_page_config(page_title="Trading Mini App", layout="wide")
st.title("📈 تطبيق تداول صغير (أمريكي + سعودي)")

# --------------------------
# ثوابت
# --------------------------
PERIOD_OPTIONS = {
    "1mo": "1mo",
    "3mo": "3mo",
    "6mo": "6mo",
    "1y": "1y",
    "2y": "2y",
    "5y": "5y",
    "max": "max",
}

# --------------------------
# دوال مساعدة
# --------------------------
def to_tadawul_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if not s:
        return s
    if s.isdigit():
        return f"{s}.SR"
    if not s.endswith(".SR"):
        return f"{s}.SR"
    return s

def normalize_symbol(symbol: str, market: str) -> str:
    s = str(symbol).strip().upper()
    if market == "سعودي (تداول)":
        return to_tadawul_symbol(s)
    return s

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _flatten_columns_if_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    # بعض الأسواق (مثل السعودي) ترجع أعمدة MultiIndex: ('Close','1120.SR')
    if df is not None and not df.empty and isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df

def analyze_symbol(symbol: str, period: str = "6mo"):
    df = yf.download(symbol, period=period, progress=False)
    df = _flatten_columns_if_multiindex(df)

    if df is None or df.empty:
        return None, f"ما قدرت أجيب بيانات للرمز: {symbol} (تأكد من الرمز والسوق)."

    if "Close" not in df.columns:
        return None, "البيانات المسترجعة ما فيها عمود Close."

    close = df["Close"].squeeze()

    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()
    df["RSI14"] = calc_rsi(close, 14)

    last_close = float(close.dropna().iloc[-1])
    last_ma20 = float(df["MA20"].dropna().iloc[-1]) if df["MA20"].dropna().shape[0] else None
    last_ma50 = float(df["MA50"].dropna().iloc[-1]) if df["MA50"].dropna().shape[0] else None
    last_rsi = float(df["RSI14"].dropna().iloc[-1]) if df["RSI14"].dropna().shape[0] else None

    trend = "غير واضح"
    if last_ma20 is not None and last_ma50 is not None:
        if last_ma20 > last_ma50:
            trend = "صاعد (MA20 فوق MA50)"
        elif last_ma20 < last_ma50:
            trend = "هابط (MA20 تحت MA50)"

    rsi_note = "—"
    if last_rsi is not None:
        if last_rsi >= 70:
            rsi_note = "تشبع شراء (RSI>=70)"
        elif last_rsi <= 30:
            rsi_note = "تشبع بيع (RSI<=30)"
        else:
            rsi_note = "طبيعي"

    summary = {
        "آخر إغلاق": last_close,
        "RSI14": last_rsi,
        "MA20": last_ma20,
        "MA50": last_ma50,
        "الاتجاه": trend,
        "حالة RSI": rsi_note,
        "عدد الأيام": int(df.shape[0]),
    }
    return df, summary

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=True, sheet_name="data")
    return output.getvalue()

# --------------------------
# واجهة التطبيق
# --------------------------
tab1, tab2, tab3 = st.tabs(["تحليل سهم واحد", "فحص قائمة أسهم", "Excel + فحص قائمة أسهم"])

# ============== تبويب 1: تحليل سهم واحد ==============
with tab1:
    colA, colB = st.columns([2, 1], gap="large")

    with colA:
        market = st.selectbox("اختر السوق", ["أمريكي", "سعودي (تداول)"], index=0)

        symbol_input = st.text_input(
            "اكتب رمز السهم (مثال: AAPL, NVDA, MSFT) أو رقم سهم سعودي مثل 1120",
            value="AAPL" if market == "أمريكي" else "1120",
        )

        period_label = st.selectbox("المدة", list(PERIOD_OPTIONS.keys()), index=2)
        period = PERIOD_OPTIONS[period_label]

        run = st.button("حلّل السهم الآن", type="primary")

    with colB:
        st.markdown("### ملاحظات سريعة")
        st.write("• للسوق السعودي: اكتب رقم السهم فقط مثل 1120 أو 2222.")
        st.write("• التطبيق يحوّل تلقائيًا إلى 1120.SR.")
        st.write("• البيانات من Yahoo Finance عبر yfinance.")

    if run:
        symbol = normalize_symbol(symbol_input, market)

        with st.spinner(f"جاري جلب البيانات لـ {symbol} ..."):
            df, summary = analyze_symbol(symbol, period)

        if df is None:
            st.error(summary)
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("آخر إغلاق", f"{summary['آخر إغلاق']:.2f}")
            c2.metric("RSI14", f"{summary['RSI14']:.2f}" if summary["RSI14"] is not None else "—")
            c3.metric("MA20", f"{summary['MA20']:.2f}" if summary["MA20"] is not None else "—")
            c4.metric("MA50", f"{summary['MA50']:.2f}" if summary["MA50"] is not None else "—")

            st.info(
                f"الاتجاه: *{summary['الاتجاه']}* | "
                f"حالة RSI: *{summary['حالة RSI']}* | "
                f"عدد الأيام: {summary['عدد الأيام']}"
            )

            # شارت
            chart_df = df[["Close", "MA20", "MA50"]].copy()
            st.line_chart(chart_df)

            # جدول
            with st.expander("عرض البيانات (آخر 30 صف)"):
                st.dataframe(df.tail(30))

            # تنزيل Excel
            excel_bytes = df_to_excel_bytes(df)
            st.download_button(
                label="تحميل البيانات Excel",
                data=excel_bytes,
                file_name=f"{symbol}_{period}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# ============== تبويب 2: فحص قائمة أسهم ==============
with tab2:
    st.write("اكتب قائمة رموز (كل رمز في سطر).")
    market2 = st.selectbox("اختر السوق (للقائمة)", ["أمريكي", "سعودي (تداول)"], index=0, key="m2")

    symbols_text = st.text_area(
        "رموز الأسهم",
        value="AAPL\nMSFT\nNVDA" if market2 == "أمريكي" else "1120\n2222\n2010",
        height=140,
    )

    period_label2 = st.selectbox("المدة", list(PERIOD_OPTIONS.keys()), index=2, key="p2")
    period2 = PERIOD_OPTIONS[period_label2]
    run2 = st.button("افحص القائمة", type="primary", key="run2")

    if run2:
        raw_symbols = [s.strip() for s in symbols_text.splitlines() if s.strip()]
        symbols = [normalize_symbol(s, market2) for s in raw_symbols]

        rows = []
        with st.spinner("جاري الفحص..."):
            for sym in symbols:
                df, summary = analyze_symbol(sym, period2)
                if df is None:
                    rows.append({"الرمز": sym, "الحالة": "فشل", "سبب": summary})
                else:
                    rows.append({
                        "الرمز": sym,
                        "الحالة": "تم",
                        "آخر إغلاق": round(summary["آخر إغلاق"], 2),
                        "RSI14": round(summary["RSI14"], 2) if summary["RSI14"] is not None else None,
                        "الاتجاه": summary["الاتجاه"],
                        "حالة RSI": summary["حالة RSI"],
                    })

        result_df = pd.DataFrame(rows)
        st.dataframe(result_df, use_container_width=True)

# ============== تبويب 3: Excel + فحص ==============
with tab3:
    st.write("ارفع ملف Excel فيه عمود اسمه: ⁠ symbol ⁠ (رمز السهم).")
    market3 = st.selectbox("اختر السوق (لملف Excel)", ["أمريكي", "سعودي (تداول)"], index=0, key="m3")

    upload = st.file_uploader("ارفع ملف Excel", type=["xlsx"])
    period_label3 = st.selectbox("المدة", list(PERIOD_OPTIONS.keys()), index=2, key="p3")
    period3 = PERIOD_OPTIONS[period_label3]
    run3 = st.button("حلّل من ملف Excel", type="primary", key="run3")

    if run3:
        if upload is None:
            st.error("ارفع ملف Excel أولاً.")
            st.stop()

        try:
            in_df = pd.read_excel(upload)
        except Exception as e:
            st.error(f"ما قدرت أقرأ الملف: {e}")
            st.stop()

        if "symbol" not in in_df.columns:
            st.error("لازم يكون فيه عمود باسم: symbol")
            st.stop()

        symbols = [normalize_symbol(s, market3) for s in in_df["symbol"].astype(str).tolist()]

        out_rows = []
        with st.spinner("جاري التحليل..."):
            for sym in symbols:
                df, summary = analyze_symbol(sym, period3)
                if df is None:
                    out_rows.append({"symbol": sym, "status": "fail", "reason": summary})
                else:
                    out_rows.append({
                        "symbol": sym,
                        "status": "ok",
                        "last_close": round(summary["آخر إغلاق"], 2),
                        "rsi14": round(summary["RSI14"], 2) if summary["RSI14"] is not None else None,
                        "trend": summary["الاتجاه"],
                        "rsi_state": summary["حالة RSI"],
                    })

        out_df = pd.DataFrame(out_rows)
        st.dataframe(out_df, use_container_width=True)

        excel_bytes = df_to_excel_bytes(out_df)
        st.download_button(
            "تحميل نتائج الفحص Excel",
            data=excel_bytes,
            file_name="scan_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )