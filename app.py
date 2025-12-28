import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("📈 Market Interaction-Based Company Performance Predictor")

tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]

st.write("Fetching historical market data...")

data = yf.download(tickers, start="2018-01-01")

prices = data["Close"]
returns = prices.pct_change()

features = pd.DataFrame(index=returns.index)

for t in tickers:
    features[f"{t}_ret"] = returns[t]
    features[f"{t}_ma5"] = returns[t].rolling(5).mean()
    features[f"{t}_ma10"] = returns[t].rolling(10).mean()
    features[f"{t}_vol5"] = returns[t].rolling(5).std()

market_avg = returns.mean(axis=1)
targets = pd.DataFrame(index=returns.index)

for t in tickers:
    targets[t] = (returns[t].shift(-1) > market_avg.shift(-1)).astype(int)

X = features.dropna()
y = targets.loc[X.index]

models = {}
for t in tickers:
    model = RandomForestClassifier()
    model.fit(X, y[t])
    models[t] = model

last = X.iloc[-1].values.reshape(1, -1)
scores = {t: models[t].predict_proba(last)[0][1] for t in tickers}

st.subheader("🏆 Predicted Ranking for Next Trading Day")

ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

for name, score in ranked:
    st.write(f"**{name}** → {round(score, 3)}")
