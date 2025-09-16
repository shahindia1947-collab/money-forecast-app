import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- Fancy CSS ---
st.markdown("""
    <style>
        .blinking-title {
            font-size: 36px;
            font-weight: bold;
            color: #00cc99;
            animation: blink 1s infinite;
            text-align: center;
            margin-bottom: 30px;
        }
        @keyframes blink {
            0% {opacity: 1;}
            50% {opacity: 0;}
            100% {opacity: 1;}
        }
        .fancy-box {
            border: 2px solid #00cc99;
            padding: 25px;
            border-radius: 15px;
            background-color: #f9fdfc;
            box-shadow: 0 0 10px rgba(0, 204, 153, 0.3);
            margin-bottom: 30px;
        }
        .watchlist-line {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .ticker-card {
            background-color: #eef;
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown('<div class="blinking-title">LET US MAKE MORE MONEY 💸</div>', unsafe_allow_html=True)

# --- Watchlist ---
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["AAPL", "MSFT", "GOOGL"]

st.markdown('<div class="fancy-box">', unsafe_allow_html=True)
st.subheader("📋 Watchlist Manager")

col1, col2 = st.columns(2)

with col1:
    new_ticker = st.text_input("Add a new ticker", placeholder="e.g. TSLA")
    if st.button("➕ Add Ticker") and new_ticker:
        ticker_upper = new_ticker.upper()
        if ticker_upper not in st.session_state.watchlist:
            st.session_state.watchlist.append(ticker_upper)
        else:
            st.warning(f"{ticker_upper} is already in your watchlist.")

with col2:
    if st.session_state.watchlist:
        remove_ticker = st.selectbox("Remove ticker", st.session_state.watchlist)
        if st.button("❌ Remove Ticker"):
            st.session_state.watchlist.remove(remove_ticker)
    else:
        st.info("Your watchlist is empty.")

# --- Display Watchlist ---
if st.session_state.watchlist:
    watchlist_html = '<div class="watchlist-line">' + ' | '.join(st.session_state.watchlist) + '</div>'
    st.markdown(watchlist_html, unsafe_allow_html=True)

    st.markdown("### 💹 Real-Time Prices")
    for ticker in st.session_state.watchlist:
        try:
            stock = yf.Ticker(ticker)
            price = stock.info.get("regularMarketPrice", "N/A")
            name = stock.info.get("shortName", ticker)
            st.markdown(f"""
                <div class="ticker-card">
                    <strong>{name} ({ticker})</strong><br>
                    Price: ${price}
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
else:
    st.markdown("Your watchlist is currently empty. Add a ticker to get started!")

st.markdown('</div>', unsafe_allow_html=True)

# --- Stock Selection ---
selected_stock = st.selectbox("📈 Select a stock to forecast", st.session_state.watchlist)

# --- Data Fetching ---
@st.cache_data
def fetch_data(ticker):
    try:
        data = yf.download(ticker, period="90d")
        if data.empty or data["Close"].dropna().empty:
            return None, None, None
        info = yf.Ticker(ticker).info
        news = yf.Ticker(ticker).news
        return data, info, news
    except Exception:
        return None, None, None

# --- Forecasting ---
def forecast_prices(data):
    try:
        df = data[['Close']].dropna()
        if len(df) < 63:
            return None

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)

        X, y = [], []
        for i in range(60, len(scaled_data) - 3):
            X.append(scaled_data[i - 60:i])
            y.append(scaled_data[i:i + 3].flatten())

        X = np.array(X).reshape((-1, 60, 1))
        y = np.array(y)

        model = Sequential([
            LSTM(50, input_shape=(60, 1)),
            Dense(3)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=10, batch_size=16, verbose=0)

        last_60 = scaled_data[-60:].reshape((1, 60, 1))
        prediction = model.predict(last_60)
        forecast = scaler.inverse_transform(prediction).flatten()
        return forecast
    except:
        return None

# --- Main Logic ---
data, info, news = fetch_data(selected_stock)

if data is None:
    st.error("Failed to load data. Try a different ticker.")
else:
    forecast = forecast_prices(data)
    last_7 = data['Close'].dropna()[-7:]
    forecast_dates = pd.date_range(start=last_7.index[-1], periods=4, freq='B')[1:]

    st.markdown('<div class="fancy-box">', unsafe_allow_html=True)
    st.subheader("📊 Price Data & Forecast")

    st.write("📌 Last 3 Closing Prices:")
    st.dataframe(last_7[-3:].reset_index().rename(columns={"Date": "Date", "Close": "Price"}))

    if forecast is not None and len(forecast) == 3:
        st.write("🔮 Forecasted Prices:")
        forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecasted Close": forecast})
        st.dataframe(forecast_df)
    else:
        st.warning("Forecast could not be generated.")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Company Info ---
    st.markdown('<div class="fancy-box">', unsafe_allow_html=True)
    st.subheader("🏢 Company Info & Highlights")
    st.write("**Business Summary:**", info.get("longBusinessSummary", "No summary available"))
    highlights = {
        "Current Price": info.get("currentPrice"),
        "Day Change": f"{info.get('regularMarketChangePercent', 0):.2f}%",
        "Volume": info.get("volume"),
        "Market Cap": info.get("marketCap"),
        "52-Week High": info.get("fiftyTwoWeekHigh"),
        "52-Week Low": info.get("fiftyTwoWeekLow")
    }
    st.table(pd.DataFrame(highlights.items(), columns=["Metric", "Value"]))
    st.markdown('</div>', unsafe_allow_html=True)

    # --- News Section ---
    st.markdown('<div class="fancy-box">', unsafe_allow_html=True)
    st.subheader("📰 Recent News")
    for article in news[:5]:
        title = article.get("title")
        link = article.get("link")
        if title and link:
            st.markdown(f"- [{title}]({link})")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Market News Button ---
    if st.button("📢 Show S&P 500 Market News"):
        sp_news = yf.Ticker("^GSPC").news
        st.markdown('<div class="fancy-box">', unsafe_allow_html=True)
        st.subheader("🌐 Market News")
        for article in sp_news[:5]:
            title = article.get("title")
            link = article.get("link")
            if title and link:
                st.markdown(f"- [{title}]({link})")
        st.markdown('</div>', unsafe_allow_html=True)