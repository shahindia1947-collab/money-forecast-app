# 💸 Money Forecast App

Welcome to the **Money Forecast App** — your AI-powered financial dashboard built with Streamlit. This app helps you manage a personalized stock watchlist, view real-time prices, forecast future stock movements using deep learning, and stay updated with the latest market news.

---

## 🚀 Features

- 📋 **Watchlist Manager**  
  Add or remove tickers and track your favorite stocks in real time.

- 💹 **Live Price Display**  
  View current market prices and company highlights for each stock.

- 🔮 **Stock Price Forecasting**  
  Uses an LSTM neural network to predict the next 3 closing prices based on recent trends.

- 📰 **News Feed**  
  Stay informed with the latest headlines for selected stocks and the S&P 500.

---

## 🧠 Tech Stack

- **Streamlit** for the interactive UI  
- **yFinance** for financial data and news  
- **TensorFlow + Keras** for LSTM-based forecasting  
- **Pandas & NumPy** for data manipulation  
- **scikit-learn** for scaling inputs

---

## 📦 Installation

To run locally:

```bash
git clone https://github.com/shahindia1947-collab/money-forecast-app.git
cd money-forecast-app
pip install -r requirements.txt
streamlit run money_forecast_app.py
