import streamlit as st
import yfinance as yf
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Trading Mini App", layout="wide")
st.title("📈 تطبيق تداول صغير (أمريكي)")

# ---- دالة RSI (بدون مكتبات إضافية) ----
def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def analyze_symbol(symbol: str, period="6mo"):
    df = yf.download(symbol, period=period, progress=False)
    if df is None or df.empty:
        return None

    close = df["Close"].squeeze()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    rsi = calc_rsi(close, 14)

    last_close = float(close.iloc[-1])
    last_ma20 = float(ma20.iloc[-1])
    last_ma50 = float(ma50.iloc[-1])
    last_rsi = float(rsi.iloc[-1])

    trend = "صاعد" if (last_close > last_ma20 and last_close > last_ma50) else \
            "هابط" if (last_close < last_ma20 and last_close < last_ma50) else "متذبذب"

    if trend == "صاعد" and last_rsi < 70:
        rec = "BUY"
    elif last_rsi >= 70:
        rec = "WAIT (تشبع شراء)"
    else:
        rec = "WAIT"

    return {
        "Symbol": symbol,
        "Last Close": round(last_close, 2),
        "MA20": round(last_ma20, 2),
        "MA50": round(last_ma50, 2),
        "RSI": round(last_rsi, 2),
        "Trend": trend,
        "Recommendation": rec
    }

# ---- واجهة التطبيق ----
tab1, tab2 = st.tabs(["تحليل سهم واحد", "فحص قائمة أسهم + Excel"])

with tab1:
    symbol = st.text_input("اكتب رمز السهم (مثال: AAPL, NVDA, MSFT)", value="AAPL").strip().upper()
    period = st.selectbox("المدة", ["3mo", "6mo", "1y", "2y"], index=1)

    if st.button("حلّل السهم الآن"):
        res = analyze_symbol(symbol, period=period)
        if not res:
            st.error("ما قدرت أجيب بيانات. تأكد من الرمز.")
        else:
            st.success(f"✅ النتيجة لـ {symbol}")
            st.json(res)

with tab2:
    st.write("اكتب الرموز مفصولة بفواصل. مثال: AAPL, MSFT, NVDA, TSLA")
    symbols_text = st.text_area("قائمة الأسهم", value="AAPL,MSFT,NVDA,TSLA")
    period2 = st.selectbox("المدة (للقائمة)", ["3mo", "6mo", "1y", "2y"], index=1, key="p2")

    if st.button("افحص القائمة"):
        symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
        results = []
        for s in symbols:
            r = analyze_symbol(s, period=period2)
            if r:
                results.append(r)

        if not results:
            st.error("ما طلع أي نتائج. راجع الرموز.")
        else:
            df_out = pd.DataFrame(results)
            st.dataframe(df_out, use_container_width=True)

            # Excel download
            bio = BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                df_out.to_excel(writer, index=False, sheet_name="Signals")
            st.download_button(
                "⬇️ تحميل النتائج Excel",
                data=bio.getvalue(),
                file_name="signals.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
