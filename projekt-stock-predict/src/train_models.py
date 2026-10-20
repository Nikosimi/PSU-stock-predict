import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBClassifier

from fetch_data import fetch_data
from indicators import add_all_ind
from config import TICKER, PERIOD, TICKERS, ML_FEATURES


# ── Data helpers ─────────────────────────────────────────────────────────────

def build_dataset(ticker, period):
    """Fetch data, add indicators, create binary target, drop NaNs."""
    data = fetch_data(ticker, period=period)
    if data is None:
        raise ValueError(f"Failed to fetch data for '{ticker}'")
    data = add_all_ind(data)
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    return data.dropna()


def make_splits(data, features=None):
    """Return X, y and chronological 80 / 10 / 10 splits."""
    if features is None:
        features = ML_FEATURES
    X = data[features]
    y = data["Target"]
    n = len(X)
    train_end = int(n * 0.80)
    val_end   = int(n * 0.90)
    return (
        X.iloc[:train_end],       y.iloc[:train_end],
        X.iloc[train_end:val_end], y.iloc[train_end:val_end],
        X.iloc[val_end:],          y.iloc[val_end:],
    )


# ── Multi-stock training set ──────────────────────────────────────────────────

def build_multi_stock_dataset(tickers, period):
    """
    Fetch multiple stocks, compute indicators for each, take the first 80%
    of each as training data, and concatenate into one combined training set.

    Validation / test / backtest are always run on the primary ticker
    separately, so there is no leakage between the splits.
    """
    frames = []
    for ticker in tickers:
        try:
            data = build_dataset(ticker, period)
            n = len(data)
            train_end = int(n * 0.80)
            frames.append(data.iloc[:train_end])
            print(f"  [{ticker}] {train_end} training rows")
        except ValueError as e:
            print(f"  [WARN] Skipping {ticker}: {e}")

    if not frames:
        raise ValueError("No data could be fetched for any ticker.")

    combined = pd.concat(frames, ignore_index=True)
    return combined[ML_FEATURES], combined["Target"]


# ── Cross-asset dataset ───────────────────────────────────────────────────────

def build_cross_asset_dataset(tickers, primary_ticker, period):
    """
    Build a cross-asset feature matrix that predicts primary_ticker's direction.

    For each trading day, the feature vector is the concatenation of ML_FEATURES
    from every ticker (e.g. AAPL_RSI_14, MSFT_RSI_14, GOOGL_RSI_14, ...).
    The model learns how correlated stocks behave on day t to predict whether
    primary_ticker will rise on day t+1.

    Returns (combined_df, cross_feature_names).
    combined_df contains all feature columns plus Target, Close, EMA_20, EMA_50
    from the primary ticker (needed for backtesting signals).
    """
    feat_frames   = {}
    primary_data  = None

    for ticker in tickers:
        try:
            raw = fetch_data(ticker, period=period)
            if raw is None:
                raise ValueError("no data returned")
            raw = add_all_ind(raw)
            feats = raw[ML_FEATURES].copy()
            feats.columns = [f"{ticker}_{col}" for col in ML_FEATURES]
            feat_frames[ticker] = feats
            if ticker == primary_ticker:
                primary_data = raw
            print(f"  [{ticker}] {len(raw)} rows")
        except Exception as e:
            print(f"  [WARN] Skipping {ticker}: {e}")

    if primary_ticker not in feat_frames or primary_data is None:
        raise ValueError(f"Primary ticker '{primary_ticker}' could not be loaded.")

    # Align all stocks by date — inner join via concatenation + dropna
    combined = pd.concat(feat_frames.values(), axis=1)

    # Attach target and backtest columns from the primary stock
    combined["Target"] = (primary_data["Close"].shift(-1) > primary_data["Close"]).astype(int)
    combined["Close"]  = primary_data["Close"]
    combined["EMA_20"] = primary_data["EMA_20"]
    combined["EMA_50"] = primary_data["EMA_50"]
    combined = combined.dropna()

    cross_features = [c for c in combined.columns
                      if c not in ("Target", "Close", "EMA_20", "EMA_50")]
    return combined, cross_features


# ── Model factory ─────────────────────────────────────────────────────────────

def get_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(64, 32), activation="relu",
                max_iter=1000, early_stopping=True, random_state=42,
            )),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced")),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_split=10,
            random_state=42, class_weight="balanced_subsample",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss",
        ),
    }


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_model(name, model, X, y, verbose=True):
    preds  = model.predict(X)
    report = classification_report(y, preds, output_dict=True)
    if verbose:
        print(f"\n{'═'*52}\n  {name}\n{'═'*52}")
        print(confusion_matrix(y, preds))
        print(classification_report(y, preds))
    return {
        "Model":           name,
        "Accuracy":        round(report["accuracy"], 3),
        "Precision (BUY)": round(report["1"]["precision"], 3),
        "Recall (BUY)":    round(report["1"]["recall"], 3),
        "F1 (BUY)":        round(report["1"]["f1-score"], 3),
    }


def sharpe_ratio(returns, trading_days=252):
    """Annualised Sharpe ratio (risk-free rate = 0)."""
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * (trading_days ** 0.5)


def max_drawdown(values):
    """Maximum peak-to-trough drawdown as a percentage."""
    peak = values.cummax()
    return ((values - peak) / peak).min() * 100


# ── Signal generation & backtest ──────────────────────────────────────────────

def generate_signals(proba, index, threshold, ema_col, data):
    """
    BUY (1) when P(up) > threshold AND close > EMA (trend confirmation).
    CASH (0) otherwise.
    """
    signals = []
    for i, p in enumerate(proba):
        close_price = data.loc[index[i], "Close"]
        ema_value   = data.loc[index[i], ema_col]
        signals.append(1 if p[1] > threshold and close_price > ema_value else 0)
    return signals


def backtest_with_hold(predictions, test_index, min_hold_days, data, initial_cash=1000):
    """Simulate a simple long/cash strategy with a minimum holding period."""
    results = data.loc[test_index].copy()
    results["Prediction"]    = predictions
    results["Market_Return"] = results["Close"].pct_change()

    position, days_in = 0, 0
    positions = []
    for signal in results["Prediction"]:
        if days_in >= min_hold_days:
            position, days_in = signal, 0
        positions.append(position)
        days_in += 1

    results["Position"]        = positions
    results["Position"]        = results["Position"].shift(1).fillna(0)
    results["Strategy_Return"] = results["Position"] * results["Market_Return"]
    results["Market_Value"]    = initial_cash * (1 + results["Market_Return"]).cumprod()
    results["Strategy_Value"]  = initial_cash * (1 + results["Strategy_Return"]).cumprod()
    return results


def generate_position_sizes(proba, index, ema_col, data):
    """
    Use P(up) directly as the fractional position size (0–1).
    Zero out when close < EMA so the trend filter still applies.
    """
    positions = []
    for i, p in enumerate(proba):
        close_price = data.loc[index[i], "Close"]
        ema_value   = data.loc[index[i], ema_col]
        positions.append(p[1] if close_price > ema_value else 0.0)
    return positions


def backtest_continuous(positions, test_index, data, initial_cash=1000):
    """Backtest with fractional position sizes; no minimum holding period."""
    results = data.loc[test_index].copy()
    results["Position"]        = pd.Series(positions, index=test_index).shift(1).fillna(0)
    results["Market_Return"]   = results["Close"].pct_change()
    results["Strategy_Return"] = results["Position"] * results["Market_Return"]
    results["Market_Value"]    = initial_cash * (1 + results["Market_Return"]).cumprod()
    results["Strategy_Value"]  = initial_cash * (1 + results["Strategy_Return"]).cumprod()
    return results


def run_grid_search(model, X_val, data, thresholds, ema_windows, min_holds):
    """Grid search on the validation set. Returns (summary_list, best_params).

    Optimises Sharpe ratio. Strategies that generate zero trades are penalised
    with Sharpe = -inf so the grid always prefers any active strategy over
    sitting in cash the entire period.
    """
    y_proba_val     = model.predict_proba(X_val)
    results_summary = []
    best_sharpe     = float("-inf")
    best_params     = None

    for threshold in thresholds:
        for ema_w in ema_windows:
            for min_hold in min_holds:
                sigs      = generate_signals(y_proba_val, X_val.index, threshold, f"EMA_{ema_w}", data)
                bt        = backtest_with_hold(sigs, X_val.index, min_hold, data)
                strat_ret = (bt["Strategy_Value"].iloc[-1] / 1000 - 1) * 100
                mkt_ret   = (bt["Market_Value"].iloc[-1]   / 1000 - 1) * 100
                n_trades  = int(sum(sigs))
                sh        = sharpe_ratio(bt["Strategy_Return"].dropna()) if n_trades > 0 else float("-inf")
                results_summary.append((threshold, ema_w, min_hold, strat_ret, mkt_ret, n_trades, sh))
                if sh > best_sharpe:
                    best_sharpe = sh
                    best_params = (threshold, ema_w, min_hold)

    return results_summary, best_params


# ── Main script ───────────────────────────────────────────────────────────────

def main():
    # ── Cross-asset dataset ────────────────────────────────────────────────
    print(f"\nBuilding cross-asset dataset: {TICKERS} → predict {TICKER}")
    data, cross_features = build_cross_asset_dataset(TICKERS, TICKER, PERIOD)
    print(f"Total rows: {len(data)}  Features: {len(cross_features)}\n")

    X_train, y_train, X_val, _, X_test, y_test = make_splits(data, cross_features)

    print(f"Primary stock ({TICKER}) class balance:\n{data['Target'].value_counts()}\n")
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # Time-series cross-validation on training portion only
    print("\n=== TIME-SERIES CROSS-VALIDATION (5 folds, training set) ===")
    tscv   = TimeSeriesSplit(n_splits=5)
    models = get_models()
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="accuracy")
        print(f"  {name:<22}  {scores.mean():.3f} ± {scores.std():.3f}")

    # Train all models on the cross-asset training set
    print("\nTraining all models on cross-asset training set...")
    for model in models.values():
        model.fit(X_train, y_train)
    print("Done.\n")

    # Evaluate on test set
    comparison_rows = []
    for name, model in models.items():
        comparison_rows.append(evaluate_model(name, model, X_test, y_test))

    comparison_df = pd.DataFrame(comparison_rows).set_index("Model")
    print("\n=== MODEL COMPARISON (test set) ===")
    print(comparison_df.to_string())

    # Feature importance grouped by stock (RF + XGBoost)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Cross-Asset Feature Importance (by stock)", fontsize=13, fontweight="bold")
    for ax, name in zip(axes, ["Random Forest", "XGBoost"]):
        imp = pd.Series(models[name].feature_importances_, index=cross_features)
        stock_imp = {t: imp[[c for c in cross_features if c.startswith(f"{t}_")]].sum()
                     for t in TICKERS}
        pd.Series(stock_imp).sort_values().plot(
            kind="barh", ax=ax, color="steelblue", edgecolor="white"
        )
        ax.set_title(name)
        ax.set_xlabel("Total Importance")
        ax.grid(axis="x", alpha=0.4)
    plt.tight_layout()
    plt.show()

    # Grid search on validation set
    thresholds  = [0.45, 0.5, 0.55, 0.6]
    ema_windows = [
        20,
        # 50,  # EMA_50 stays elevated for months after crashes, blocking recovery signals
    ]
    min_holds   = [1, 3, 5]
    xgb_model   = models["XGBoost"]

    results_summary, best_params = run_grid_search(
        xgb_model, X_val, data, thresholds, ema_windows, min_holds
    )

    print("\n=== GRID SEARCH (validation set) ===")
    for r in results_summary:
        print(f"  thr={r[0]}  EMA={r[1]}  hold={r[2]}  "
              f"strategy={r[3]:.2f}%  market={r[4]:.2f}%  trades={r[5]}  sharpe={r[6]:.3f}")

    thresh, ema_w, min_hold = best_params
    best_sharpe = max(r[6] for r in results_summary if r[6] != float("-inf"))
    print(f"\nBest params: thr={thresh}  EMA={ema_w}  hold={min_hold}  "
          f"val_sharpe={best_sharpe:.3f}")

    # Apply best params to test set
    y_proba_test     = xgb_model.predict_proba(X_test)
    final_signals    = generate_signals(y_proba_test, X_test.index, thresh, f"EMA_{ema_w}", data)
    backtest_results = backtest_with_hold(final_signals, X_test.index, min_hold, data)

    total_market   = (backtest_results["Market_Value"].iloc[-1]   / 1000 - 1) * 100
    total_strategy = (backtest_results["Strategy_Value"].iloc[-1] / 1000 - 1) * 100
    sharpe_strat   = sharpe_ratio(backtest_results["Strategy_Return"].dropna())
    sharpe_market  = sharpe_ratio(backtest_results["Market_Return"].dropna())
    dd_strat       = max_drawdown(backtest_results["Strategy_Value"])
    dd_market      = max_drawdown(backtest_results["Market_Value"])

    print("\n=== TEST SET BACKTEST (best params) ===")
    print(f"{'':25} {'Strategy':>12} {'Market':>12}")
    print(f"{'Total return':25} {total_strategy:>11.2f}% {total_market:>11.2f}%")
    print(f"{'Sharpe ratio':25} {sharpe_strat:>12.3f} {sharpe_market:>12.3f}")
    print(f"{'Max drawdown':25} {dd_strat:>11.2f}% {dd_market:>11.2f}%")
    print(backtest_results[[
        "Close", "Prediction", "Market_Return", "Strategy_Return",
        "Market_Value", "Strategy_Value",
    ]].tail())

    # ── Position sizing comparison ─────────────────────────────────────────
    pos_sizes = generate_position_sizes(y_proba_test, X_test.index, f"EMA_{ema_w}", data)
    bt_ps     = backtest_continuous(pos_sizes, X_test.index, data)
    total_ps  = (bt_ps["Strategy_Value"].iloc[-1] / 1000 - 1) * 100
    sharpe_ps = sharpe_ratio(bt_ps["Strategy_Return"].dropna())
    dd_ps     = max_drawdown(bt_ps["Strategy_Value"])

    print("\n=== STRATEGY COMPARISON (test set) ===")
    print(f"{'':25} {'Pos.Sizing':>12} {'Binary':>12} {'Market':>12}")
    print(f"{'Total return':25} {total_ps:>11.2f}% {total_strategy:>11.2f}% {total_market:>11.2f}%")
    print(f"{'Sharpe ratio':25} {sharpe_ps:>12.3f} {sharpe_strat:>12.3f} {sharpe_market:>12.3f}")
    print(f"{'Max drawdown':25} {dd_ps:>11.2f}% {dd_strat:>11.2f}% {dd_market:>11.2f}%")

    # Plot
    buy_points  = backtest_results[backtest_results["Prediction"] == 1]
    cash_points = backtest_results[backtest_results["Prediction"] == 0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(
        f"XGBoost — Strategy vs Market (Test Set)\n"
        f"Binary: thr={thresh}  EMA={ema_w}  hold={min_hold}d  "
        f"Sharpe={sharpe_strat:.2f}  MaxDD={dd_strat:.1f}%  |  "
        f"Pos.Sizing: Sharpe={sharpe_ps:.2f}  MaxDD={dd_ps:.1f}%",
        fontsize=10, fontweight="bold",
    )
    axes[0].plot(backtest_results.index, backtest_results["Market_Value"],
                 label="Market (Buy & Hold)", color="steelblue")
    axes[0].plot(backtest_results.index, backtest_results["Strategy_Value"],
                 label="Binary Strategy", color="darkorange")
    axes[0].plot(bt_ps.index, bt_ps["Strategy_Value"],
                 label="Position Sizing", color="forestgreen", linestyle="--")
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    axes[1].plot(backtest_results.index, backtest_results["Close"],
                 color="grey", linewidth=0.8, label="Close Price")
    axes[1].scatter(buy_points.index,  buy_points["Close"],
                    color="green", s=12, label="BUY",  zorder=5)
    axes[1].scatter(cash_points.index, cash_points["Close"],
                    color="red",   s=12, label="CASH", zorder=5)
    axes[1].set_ylabel("Price ($)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    axes[2].bar(
        backtest_results.index,
        backtest_results["Strategy_Return"],
        color=np.where(backtest_results["Strategy_Return"] >= 0, "forestgreen", "tomato"),
        width=1, alpha=0.7, label="Daily Strategy Return",
    )
    axes[2].axhline(0, color="black", linewidth=0.6)
    axes[2].set_ylabel("Daily Return")
    axes[2].set_xlabel("Date")
    axes[2].legend(fontsize=8)
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
