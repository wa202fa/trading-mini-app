import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Trading App (Clean)", layout="wide")

# =========================
# Session State
# =========================
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# =========================
# Sidebar
# =========================
st.sidebar.title("📌 القوائم")

market = st.sidebar.selectbox("السوق", ["🇺🇸 أمريكا", "🇸🇦 السعودية"])
period = st.sidebar.selectbox("اختر المدة", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
top_n = st.sidebar.selectbox("كم سهم نعرض في الترتيب", [5, 10, 15, 20], index=2)

symbol_input = st.sidebar.text_input("ابحث (رمز السهم)")

col1, col2 = st.sidebar.columns(2)

# =========================
# Add to watchlist
# =========================
if col1.button("➕ أضف للسلة"):
    sym = symbol_input.strip().upper()

    # تحويل السعودي تلقائيًا إلى .SR
    if market == "🇸🇦 السعودية" and sym.isdigit():
        sym = f"{sym}.SR"

    if sym and sym not in st.session_state.watchlist:
        st.session_state.watchlist.append(sym)

if col2.button("🗑️ مسح السلة"):
    st.session_state.watchlist = []

# =========================
# Show watchlist
# =========================
st.sidebar.subheader("🧺 سلة المتابعة")
if st.session_state.watchlist:
    for s in st.session_state.watchlist:
        st.sidebar.write(f"• {s}")
else:
    st.sidebar.write("فاضية")

# =========================
# Main
# =========================
st.title("🔍 فحص قائمة المتابعة — نسخة نظيفة")

st.write(f"عدد الأسهم في السلة: *{len(st.session_state.watchlist)}*")

if st.button("🚀 افحص السلة"):
    if not st.session_state.watchlist:
        st.warning("السلة فاضية!")
    else:
        rows = []

        for sym in st.session_state.watchlist:
            try:
                ticker = yf.Ticker(sym)
                df = ticker.history(period=period)

                if df.empty:
                    rows.append({
                        "الرمز": sym,
                        "السعر الأخير": None,
                        "التغير %": None,
                        "الحالة": "❌ لا توجد بيانات"
                    })
                    continue

                last_close = df["Close"].iloc[-1]
                prev_close = df["Close"].iloc[-2] if len(df) > 1 else last_close
                change_pct = ((last_close - prev_close) / prev_close) * 100 if prev_close else 0

                rows.append({
                    "الرمز": sym,
                    "السعر الأخير": round(float(last_close), 2),
                    "التغير %": round(float(change_pct), 2),
                    "الحالة": "✅ تم التحليل"
                })

            except Exception:
                rows.append({
                    "الرمز": sym,
                    "السعر الأخير": None,
                    "التغير %": None,
                    "الحالة": "❌ خطأ"
                })

        out = pd.DataFrame(rows)

        st.subheader("📊 النتائج")
        st.dataframe(out, use_container_width=True)

        # ترتيب حسب أعلى تغير
        out_sorted = out.sort_values(by="التغير %", ascending=False, na_position="last")

        st.subheader("🏆 الأعلى تغيير")
        st.dataframe(out_sorted.head(top_n), use_container_width=True)
