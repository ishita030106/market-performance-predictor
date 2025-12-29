import yfinance as yf
import pandas as pd

tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]

data = yf.download(tickers, start="2018-01-01", end="2024-12-01")

print(data.head())
print(data.columns)

prices = data["Close"]
volume = data["Volume"]

print(prices.head())
# --- DAILY RETURNS ---
returns = prices.pct_change()

print("\nDaily returns:")
print(returns.head())
features = pd.DataFrame(index=returns.index)

for t in tickers:
    features[f"{t}_ret"] = returns[t]
    features[f"{t}_ma5"] = returns[t].rolling(5).mean()
    features[f"{t}_ma10"] = returns[t].rolling(10).mean()
    features[f"{t}_vol5"] = returns[t].rolling(5).std()

print("\nFeatures preview:")
print(features.head())
sector_map = {
    "AAPL": "tech",
    "MSFT": "tech",
    "GOOGL": "tech",
    "JPM": "finance",
    "XOM": "energy"
}

features["tech_avg"] = returns[["AAPL", "MSFT", "GOOGL"]].mean(axis=1)
features["finance_avg"] = returns[["JPM"]].mean(axis=1)
features["energy_avg"] = returns[["XOM"]].mean(axis=1)

print("\nWith interaction features:")
print(features.head())
market_avg = returns.mean(axis=1)

targets = pd.DataFrame(index=returns.index)

for t in tickers:
    targets[t] = (returns[t].shift(-1) > market_avg.shift(-1)).astype(int)

print("\nTargets preview:")
print(targets.head())
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = features.dropna()
y = targets.loc[X.index]

train_X, test_X, train_y, test_y = train_test_split(X, y, shuffle=False)

models = {}

for t in tickers:
    clf = RandomForestClassifier(n_estimators=120, random_state=42)
    clf.fit(train_X, train_y[t])
    models[t] = clf

print("\nModels trained successfully!")
last_row = X.iloc[-1].values.reshape(1, -1)

scores = {}

for t in tickers:
    scores[t] = models[t].predict_proba(last_row)[0][1]

ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)

print("\nPredicted ranking for next day:")
for name, score in ranking:
    print(name, "->", round(score, 3))
import pandas as pd

for t in tickers:
    importances = pd.Series(models[t].feature_importances_, index=X.columns)
    print(f"\nTop features influencing {t}:")
    print(importances.sort_values(ascending=False).head(5))
corr = returns.corr()
print("\nCompany correlation matrix:")
print(corr)

