import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4

from io import BytesIO
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="TradeIQ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# PREMIUM CSS
# =========================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b1220 0%, #111827 50%, #0f172a 100%);
        color: white;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0px;
        letter-spacing: 0.3px;
    }

    .sub-title {
        color: #94a3b8;
        font-size: 16px;
        margin-top: -6px;
        margin-bottom: 18px;
    }

    .hero-box {
        background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(99,102,241,0.10));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 28px 28px 20px 28px;
        box-shadow: 0px 10px 35px rgba(0,0,0,0.22);
        margin-bottom: 20px;
    }

    .glass-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 18px;
        box-shadow: 0px 8px 30px rgba(0,0,0,0.22);
        margin-bottom: 18px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 10px;
        color: #ffffff;
    }

    .section-sub {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 16px;
    }

    .mini-tag {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.08);
        color: #e2e8f0;
        font-size: 12px;
        margin-right: 8px;
        margin-top: 6px;
    }

    .signal-buy {
        padding: 10px 14px;
        border-radius: 14px;
        background: rgba(34,197,94,0.15);
        border: 1px solid rgba(34,197,94,0.25);
        color: #86efac;
        font-weight: 600;
        text-align: center;
    }

    .signal-sell {
        padding: 10px 14px;
        border-radius: 14px;
        background: rgba(239,68,68,0.15);
        border: 1px solid rgba(239,68,68,0.25);
        color: #fca5a5;
        font-weight: 600;
        text-align: center;
    }

    .signal-hold {
        padding: 10px 14px;
        border-radius: 14px;
        background: rgba(234,179,8,0.15);
        border: 1px solid rgba(234,179,8,0.25);
        color: #fde68a;
        font-weight: 600;
        text-align: center;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.07);
        padding: 14px;
        border-radius: 18px;
        box-shadow: 0px 6px 18px rgba(0,0,0,0.18);
    }

    .footer-note {
        color: #64748b;
        font-size: 12px;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER / HERO
# =========================
st.markdown("""
<div class="hero-box">
    <div class="main-title">📊 TradeIQ</div>
    <div class="sub-title">AI-powered stock analysis, prediction, model evaluation, and portfolio simulation dashboard</div>
    <span class="mini-tag">Machine Learning</span>
    <span class="mini-tag">Technical Indicators</span>
    <span class="mini-tag">Prediction Engine</span>
    <span class="mini-tag">Portfolio Insights</span>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("## TradeIQ")

stock = st.sidebar.selectbox(
    "Select Stock",
    ["TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX"]
)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Prediction", "Charts", "Model Analysis", "Portfolio", "History"]
)

predict_date = st.sidebar.date_input("Prediction Date", pd.to_datetime("today"))

# =========================
# LOAD DATA
# =========================
data = yf.download(stock, start=start_date, end=end_date)

if data.empty:
    st.error("No data found. Please change stock or date range.")
    st.stop()

data.reset_index(inplace=True)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

if 'Date' not in data.columns:
    data = data.reset_index()

required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in required_cols:
    if col not in data.columns:
        st.error(f"Required column '{col}' not found.")
        st.stop()

data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
for col in required_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data.dropna(subset=['Date'] + required_cols, inplace=True)
data = data.sort_values("Date").reset_index(drop=True)

# =========================
# TECHNICAL INDICATORS
# =========================
data['Daily Change'] = data['Close'] - data['Open']
data['Return'] = data['Close'].pct_change()

data['MA10'] = data['Close'].rolling(10, min_periods=1).mean()
data['MA30'] = data['Close'].rolling(30, min_periods=1).mean()
data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()

data['BB_Middle'] = data['Close'].rolling(20, min_periods=1).mean()
data['BB_STD'] = data['Close'].rolling(20, min_periods=1).std()
data['BB_Upper'] = data['BB_Middle'] + (2 * data['BB_STD'])
data['BB_Lower'] = data['BB_Middle'] - (2 * data['BB_STD'])

delta = data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(window=14, min_periods=1).mean()
avg_loss = loss.rolling(window=14, min_periods=1).mean()

rs = avg_gain / avg_loss.replace(0, np.nan)
data['RSI'] = 100 - (100 / (1 + rs))
data['RSI'] = data['RSI'].fillna(50)

data = data.bfill().ffill()

# =========================
# MODEL DATA
# =========================
feature_cols = ['Open', 'High', 'Low', 'Volume', 'MA10', 'MA30', 'EMA20', 'RSI']
X = data[feature_cols]
y = data['Close']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42)
}

results = []
trained_models = {}

for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    preds = mdl.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append({
        "Model": name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2 Score": round(r2, 4)
    })

    trained_models[name] = mdl

results_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False).reset_index(drop=True)
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

# =========================
# CURRENT PREDICTION
# =========================
latest = data.iloc[-1]

input_df = pd.DataFrame([[
    latest['Open'],
    latest['High'],
    latest['Low'],
    latest['Volume'],
    latest['MA10'],
    latest['MA30'],
    latest['EMA20'],
    latest['RSI']
]], columns=feature_cols)

input_scaled = scaler.transform(input_df)
pred_value = float(best_model.predict(input_scaled).flatten()[0])

current_price = float(latest['Close'])
change = float(pred_value - current_price)
percent = float((change / current_price) * 100)

if percent > 1:
    signal = "BUY"
elif percent < -1:
    signal = "SELL"
else:
    signal = "HOLD"

# =========================
# HISTORY
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

new_entry = {
    "Prediction Date": str(predict_date),
    "Stock": stock,
    "Best Model": best_model_name,
    "Predicted Price": round(pred_value, 2),
    "Actual Price": round(current_price, 2),
    "Change %": round(percent, 2),
    "Signal": signal,
    "Generated At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

if len(st.session_state.history) == 0 or st.session_state.history[-1] != new_entry:
    st.session_state.history.append(new_entry)

history_df = pd.DataFrame(st.session_state.history)

# =========================
# SIGNAL UI
# =========================
if signal == "BUY":
    signal_html = f'<div class="signal-buy">📈 Signal: {signal}</div>'
elif signal == "SELL":
    signal_html = f'<div class="signal-sell">📉 Signal: {signal}</div>'
else:
    signal_html = f'<div class="signal-hold">⚖️ Signal: {signal}</div>'

# =========================
# TOP KPI STRIP
# =========================
st.markdown("## 📌 Market Snapshot")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Open", f"${latest['Open']:.2f}")
c2.metric("High", f"${latest['High']:.2f}")
c3.metric("Low", f"${latest['Low']:.2f}")
c4.metric("Close", f"${latest['Close']:.2f}")
c5.metric("Volume", f"{int(latest['Volume']):,}")
c6.metric("Predicted", f"${pred_value:.2f}", f"{percent:.2f}%")

st.markdown(signal_html, unsafe_allow_html=True)
st.markdown("---")

# =========================
# DASHBOARD PAGE
# =========================
if page == "Dashboard":
    st.markdown("## Dashboard Overview")

    left, right = st.columns([2.3, 1])

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Price Trend Overview</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Live trend with short-term and medium-term moving averages</div>', unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['Date'], y=data['Close'],
            name="Close Price", mode='lines',
            line=dict(width=3)
        ))
        fig.add_trace(go.Scatter(
            x=data['Date'], y=data['MA10'],
            name="MA10", mode='lines',
            line=dict(width=2, dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=data['Date'], y=data['MA30'],
            name="MA30", mode='lines',
            line=dict(width=2, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=data['Date'], y=data['EMA20'],
            name="EMA20", mode='lines',
            line=dict(width=2)
        ))

        fig.update_layout(
            title=f"{stock} Price Movement",
            template="plotly_dark",
            height=520,
            xaxis_title="Date",
            yaxis_title="Price",
            legend_title="Indicators",
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction Summary</div>', unsafe_allow_html=True)
        st.metric("Best Model", best_model_name)
        st.metric("Predicted Price", f"${pred_value:.2f}")
        st.metric("Current Price", f"${current_price:.2f}")
        st.metric("RSI", f"{latest['RSI']:.2f}")
        st.metric("Prediction Date", str(predict_date))
        st.markdown(signal_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    b1, b2 = st.columns([1.5, 1])

    with b1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Bollinger Bands Analysis</div>', unsafe_allow_html=True)
        bb_fig = go.Figure()
        bb_fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close'))
        bb_fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Upper'], name='Upper Band'))
        bb_fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Middle'], name='Middle Band'))
        bb_fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Lower'], name='Lower Band'))
        bb_fig.update_layout(
            title="Volatility Range Visualization",
            template="plotly_dark",
            height=430,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(bb_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with b2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Model Highlights</div>', unsafe_allow_html=True)
        top_row = results_df.iloc[0]
        st.metric("R² Score", top_row['R2 Score'])
        st.metric("MAE", top_row['MAE'])
        st.metric("RMSE", top_row['RMSE'])

        if latest['RSI'] > 70:
            st.warning("RSI suggests overbought conditions.")
        elif latest['RSI'] < 30:
            st.info("RSI suggests oversold conditions.")
        else:
            st.success("RSI is currently in a neutral zone.")

        st.markdown('<p class="footer-note">Technical indicators are supportive tools and should not be used as the only trading decision basis.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PREDICTION PAGE
# =========================
elif page == "Prediction":
    st.markdown("## Prediction Analysis")

    p1, p2, p3 = st.columns(3)
    p1.metric("Predicted Price", f"${pred_value:.2f}", f"{percent:.2f}%")
    p2.metric("Current Price", f"${current_price:.2f}")
    p3.metric("Signal", signal)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Actual vs Predicted</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Model behavior across recent market movement</div>', unsafe_allow_html=True)

    compare_df = data.tail(60).copy()
    compare_scaled = scaler.transform(compare_df[feature_cols])
    compare_df['Predicted'] = best_model.predict(compare_scaled)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=compare_df['Date'], y=compare_df['Close'],
        name="Actual", line=dict(width=3)
    ))
    fig2.add_trace(go.Scatter(
        x=compare_df['Date'], y=compare_df['Predicted'],
        name="Predicted", line=dict(width=3, dash='dash')
    ))

    fig2.update_layout(
        title="Actual vs Predicted Stock Price",
        template="plotly_dark",
        height=520,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prediction Error Table</div>', unsafe_allow_html=True)
    error_df = compare_df[['Date', 'Close', 'Predicted']].copy()
    error_df['Error'] = error_df['Close'] - error_df['Predicted']
    st.dataframe(error_df.tail(20), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# CHARTS PAGE
# =========================
elif page == "Charts":
    st.markdown("## Technical Charts")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Candlestick + Moving Averages</div>', unsafe_allow_html=True)
    fig_candle = go.Figure(data=[go.Candlestick(
        x=data['Date'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick'
    )])

    fig_candle.add_trace(go.Scatter(x=data['Date'], y=data['MA10'], mode='lines', name='MA10'))
    fig_candle.add_trace(go.Scatter(x=data['Date'], y=data['MA30'], mode='lines', name='MA30'))
    fig_candle.add_trace(go.Scatter(x=data['Date'], y=data['EMA20'], mode='lines', name='EMA20'))

    fig_candle.update_layout(
        title=f"{stock} Candlestick Chart",
        template="plotly_dark",
        height=680,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    st.plotly_chart(fig_candle, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">RSI Indicator</div>', unsafe_allow_html=True)
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], name='RSI', line=dict(width=3)))
        rsi_fig.add_hline(y=70, line_dash="dash")
        rsi_fig.add_hline(y=30, line_dash="dash")
        rsi_fig.update_layout(
            title="Relative Strength Index",
            template="plotly_dark",
            height=420,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(rsi_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Trading Volume</div>', unsafe_allow_html=True)
        vol_fig = go.Figure()
        vol_fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name='Volume'))
        vol_fig.update_layout(
            title="Volume Analysis",
            template="plotly_dark",
            height=420,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(vol_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# MODEL ANALYSIS PAGE
# =========================
elif page == "Model Analysis":
    st.markdown("## Model Analysis")

    mleft, mright = st.columns([1.5, 1])

    with mleft:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Model Comparison Table</div>', unsafe_allow_html=True)
        st.dataframe(results_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with mright:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Best Model Summary</div>', unsafe_allow_html=True)
        st.success(f"Best Model: {best_model_name}")
        best_row = results_df.iloc[0]
        st.metric("MAE", best_row['MAE'])
        st.metric("RMSE", best_row['RMSE'])
        st.metric("R² Score", best_row['R2 Score'])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">R² Score Comparison</div>', unsafe_allow_html=True)
    model_fig = go.Figure()
    model_fig.add_trace(go.Bar(x=results_df['Model'], y=results_df['R2 Score'], name='R2 Score'))
    model_fig.update_layout(
        title="Model Performance Comparison",
        template="plotly_dark",
        height=450,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    st.plotly_chart(model_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PORTFOLIO PAGE
# =========================
elif page == "Portfolio":
    st.markdown("## Portfolio Calculator")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Profit / Loss Estimator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Estimate outcome based on your capital and expected sell price</div>', unsafe_allow_html=True)

    invest_amount = st.number_input("Investment Amount ($)", min_value=0.0, value=1000.0, step=100.0)
    buy_price = st.number_input("Buy Price ($)", min_value=0.0, value=current_price, step=1.0)
    sell_price = st.number_input("Expected Sell Price ($)", min_value=0.0, value=pred_value, step=1.0)

    if buy_price > 0:
        shares = invest_amount / buy_price
        final_value = shares * sell_price
        profit_loss = final_value - invest_amount
        return_pct = (profit_loss / invest_amount) * 100 if invest_amount > 0 else 0

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Shares", f"{shares:.2f}")
        p2.metric("Final Value", f"${final_value:.2f}")
        p3.metric("Profit / Loss", f"${profit_loss:.2f}")
        p4.metric("Return %", f"{return_pct:.2f}%")

        if profit_loss > 0:
            st.success("Projected trade appears profitable.")
        elif profit_loss < 0:
            st.error("Projected trade indicates a loss.")
        else:
            st.info("Projected trade is neutral.")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# HISTORY PAGE
# =========================
elif page == "History":
    st.markdown("## Prediction History")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if history_df.empty:
        st.info("No prediction history available yet.")
    else:
        st.dataframe(history_df, use_container_width=True)

        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Prediction History CSV",
            data=csv,
            file_name="tradeiq_prediction_history.csv",
            mime="text/csv"
        )
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PDF CHART GENERATOR
# =========================
def create_pdf_charts(data):
    charts = []

    # Closing Price Chart
    buf1 = BytesIO()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data['Date'], data['Close'], linewidth=2)
    ax.set_title("Closing Price Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(buf1, format='png', dpi=150)
    plt.close(fig)
    buf1.seek(0)
    charts.append(buf1)

    # Volume Chart
    buf2 = BytesIO()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(data['Date'], data['Volume'])
    ax.set_title("Volume Analysis")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(buf2, format='png', dpi=150)
    plt.close(fig)
    buf2.seek(0)
    charts.append(buf2)

    return charts

# =========================
# PREMIUM PDF REPORT
# =========================
st.sidebar.markdown("---")
if st.sidebar.button("📄 Download Premium Report", use_container_width=True):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)

    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("<b>TradeIQ - Stock Analysis Report</b>", styles['Title']))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Paragraph(f"Stock: {stock}", styles['Normal']))
    elements.append(Paragraph(f"Prediction Date: {predict_date}", styles['Normal']))
    elements.append(Spacer(1, 15))

    # Market Summary
    summary_data = [
        ["Metric", "Value"],
        ["Open", f"${latest['Open']:.2f}"],
        ["High", f"${latest['High']:.2f}"],
        ["Low", f"${latest['Low']:.2f}"],
        ["Close", f"${latest['Close']:.2f}"],
        ["Volume", f"{int(latest['Volume']):,}"],
    ]

    table = Table(summary_data, colWidths=[220, 220])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))

    elements.append(Paragraph("<b>Market Summary</b>", styles['Heading2']))
    elements.append(Spacer(1, 8))
    elements.append(table)
    elements.append(Spacer(1, 20))

    # Prediction Summary
    pred_data = [
        ["Metric", "Value"],
        ["Best Model", best_model_name],
        ["Predicted Price", f"${pred_value:.2f}"],
        ["Current Price", f"${current_price:.2f}"],
        ["Change (%)", f"{percent:.2f}%"],
        ["Signal", signal],
    ]

    pred_table = Table(pred_data, colWidths=[220, 220])
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))

    elements.append(Paragraph("<b>Prediction Summary</b>", styles['Heading2']))
    elements.append(Spacer(1, 8))
    elements.append(pred_table)
    elements.append(Spacer(1, 20))

    # Model Interpretation
    if percent > 1:
        summary = f"The model predicts a potential upward movement of {percent:.2f}%."
    elif percent < -1:
        summary = f"The model predicts a potential downward movement of {abs(percent):.2f}%."
    else:
        summary = "The model predicts minimal movement in price."

    elements.append(Paragraph("<b>Model Interpretation</b>", styles['Heading2']))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(summary, styles['Normal']))
    elements.append(Spacer(1, 20))

    # Add Charts
    elements.append(Paragraph("<b>Visual Analysis</b>", styles['Heading2']))
    elements.append(Spacer(1, 10))

    charts = create_pdf_charts(data)

    for chart in charts:
        img = RLImage(chart, width=500, height=220)
        elements.append(img)
        elements.append(Spacer(1, 15))

    # Footer
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(
        "Note: This report is generated using machine learning models and technical indicators. It should not be considered as financial advice.",
        styles['Normal']
    ))

    # Build PDF
    doc.build(elements)

    pdf = buffer.getvalue()
    buffer.close()

    st.sidebar.success("✅ Premium PDF Generated!")
    st.sidebar.download_button(
        "⬇️ Download Report",
        pdf,
        "TradeIQ_Premium_Report.pdf",
        mime="application/pdf"
    )