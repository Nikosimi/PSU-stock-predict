import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from config import ML_FEATURES, TICKERS
from train_models import (
    build_dataset, build_cross_asset_dataset, make_splits, get_models,
    evaluate_model, sharpe_ratio, max_drawdown, generate_signals,
    backtest_with_hold, run_grid_search,
    generate_position_sizes, backtest_continuous,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")
st.title("Stock Price Prediction")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker symbol", value="AAPL").upper().strip()
    period = st.selectbox("Period", ["1y", "2y", "5y", "10y"], index=3)
    run_btn = st.button("▶ Run", type="primary", use_container_width=True)

if run_btn:
    st.session_state["ticker"] = ticker
    st.session_state["period"] = period

if "ticker" not in st.session_state:
    st.info("Set a ticker and period in the sidebar, then click **Run**.")
    st.stop()

ticker = st.session_state["ticker"]
period = st.session_state["period"]

# ── Cached data & models ──────────────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching & preparing data…")
def load_data(ticker, period):
    return build_dataset(ticker, period)


@st.cache_data(show_spinner="Building cross-asset feature set…")
def load_cross_asset(ticker, period):
    return build_cross_asset_dataset(TICKERS, ticker, period)


@st.cache_resource(show_spinner="Training models on cross-asset features…")
def get_trained_models(ticker, period):
    cross_data, cross_features = load_cross_asset(ticker, period)
    X_train, y_train, _, _, X_test, y_test = make_splits(cross_data, cross_features)

    tscv   = TimeSeriesSplit(n_splits=5)
    models = get_models()

    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="accuracy")
        cv_results[name] = {"mean": scores.mean(), "std": scores.std()}

    for model in models.values():
        model.fit(X_train, y_train)

    comparison_rows = [
        evaluate_model(name, model, X_test, y_test, verbose=False)
        for name, model in models.items()
    ]
    comparison_df = pd.DataFrame(comparison_rows).set_index("Model")

    return models, cv_results, comparison_df


try:
    data = load_data(ticker, period)
    cross_data, cross_features = load_cross_asset(ticker, period)
except ValueError as e:
    st.error(str(e))
    st.stop()

_, _, X_val, _, X_test, y_test = make_splits(cross_data, cross_features)
models, cv_results, comparison_df = get_trained_models(ticker, period)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["EDA", "Models", "Backtest", "Live Signal"])


# ════════════════════════════════════════════════════════════════════════════
# Tab 1 — EDA
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"{ticker} — Exploratory Data Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Trading days", len(data))
    col2.metric("Date range", f"{data.index[0].date()} → {data.index[-1].date()}")
    col3.metric("BUY days (%)", f"{data['Target'].mean() * 100:.1f}%")

    # Price + RSI + MACD
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=["Price & Indicators", "RSI (14)", "MACD"],
        vertical_spacing=0.06,
    )

    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close",
                             line=dict(color="steelblue", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["SMA_20"], name="SMA 20",
                             line=dict(color="orange", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["EMA_50"], name="EMA 50",
                             line=dict(color="red", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["Upper_band_20"],
                             line=dict(color="rgba(128,128,128,0.3)", width=0),
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["Lower_band_20"], name="Bollinger Bands",
                             fill="tonexty", fillcolor="rgba(128,128,128,0.1)",
                             line=dict(color="rgba(128,128,128,0.3)", width=0)), row=1, col=1)

    fig.add_trace(go.Scatter(x=data.index, y=data["RSI_14"], name="RSI",
                             line=dict(color="darkorange", width=1), showlegend=False), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red",   line_width=0.8, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=0.8, row=2, col=1)

    fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD",
                             line=dict(color="steelblue", width=1), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["Signal_Line"], name="Signal",
                             line=dict(color="tomato", width=1), showlegend=False), row=3, col=1)
    hist_colors = ["forestgreen" if v >= 0 else "tomato" for v in data["MACD_Histogram"]]
    fig.add_trace(go.Bar(x=data.index, y=data["MACD_Histogram"], name="Histogram",
                         marker_color=hist_colors, showlegend=False), row=3, col=1)

    fig.update_layout(height=680, legend=dict(orientation="h", y=1.04),
                      margin=dict(t=60, b=20))
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap | Return distribution
    col_l, col_r = st.columns(2)

    with col_l:
        st.write("**Feature Correlation Matrix**")
        corr = data[ML_FEATURES + ["Target"]].corr().round(2)
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, aspect="auto")
        fig_corr.update_layout(height=420, margin=dict(t=20, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_r:
        st.write("**Daily Returns Distribution**")
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Histogram(x=data["Daily_Return"], nbinsx=80,
                                       marker_color="steelblue", opacity=0.75))
        fig_ret.add_vline(x=0, line_dash="dash", line_color="red")
        fig_ret.update_layout(height=420, margin=dict(t=20, b=20),
                              xaxis_title="Daily Return (%)", yaxis_title="Count",
                              showlegend=False)
        st.plotly_chart(fig_ret, use_container_width=True)

    # Class balance pie
    st.write("**Target Class Balance**")
    counts = data["Target"].value_counts().sort_index()
    fig_pie = px.pie(values=counts.values, names=["SELL / HOLD (0)", "BUY (1)"],
                     color_discrete_sequence=["tomato", "forestgreen"])
    fig_pie.update_layout(height=280, margin=dict(t=20, b=20))
    _, c, _ = st.columns([1, 1, 1])
    c.plotly_chart(fig_pie, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# Tab 2 — Models
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Training & Evaluation")

    st.write("**Time-Series Cross-Validation (5 folds, training set only)**")
    cv_df = pd.DataFrame([
        {"Model": name, "CV Accuracy": f"{v['mean']:.3f}", "± Std": f"±{v['std']:.3f}"}
        for name, v in cv_results.items()
    ]).set_index("Model")
    st.dataframe(cv_df, use_container_width=True)

    st.write("**Model Comparison (test set)**")
    st.dataframe(
        comparison_df.style.highlight_max(color="lightgreen", axis=0),
        use_container_width=True,
    )

    st.write("**Feature Importance by Stock (cross-asset)**")
    col_rf, col_xgb = st.columns(2)
    for col, name in zip([col_rf, col_xgb], ["Random Forest", "XGBoost"]):
        imp = pd.Series(models[name].feature_importances_, index=cross_features)
        stock_imp = pd.Series({
            t: imp[[c for c in cross_features if c.startswith(f"{t}_")]].sum()
            for t in TICKERS
        }).sort_values()
        fig_imp = go.Figure(go.Bar(
            x=stock_imp.values, y=stock_imp.index,
            orientation="h", marker_color="steelblue",
        ))
        fig_imp.update_layout(title=name, height=360,
                              xaxis_title="Total Importance",
                              margin=dict(t=40, b=20))
        col.plotly_chart(fig_imp, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# Tab 3 — Backtest
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Backtest — XGBoost Strategy")

    thresholds  = [0.45, 0.5, 0.55, 0.6]
    ema_windows = [
        20,
        # 50,  # EMA_50 stays elevated for months after crashes, blocking recovery signals
    ]
    min_holds   = [1, 3, 5]
    xgb_model   = models["XGBoost"]

    results_summary, best_params = run_grid_search(
        xgb_model, X_val, cross_data, thresholds, ema_windows, min_holds
    )
    thresh, ema_w, min_hold = best_params

    with st.expander("Grid search results (validation set)"):
        gs_df = pd.DataFrame(results_summary,
                             columns=["Threshold", "EMA", "Hold (days)",
                                      "Strategy %", "Market %", "# Trades", "Sharpe"])
        st.dataframe(gs_df, use_container_width=True)

    st.info(f"Best params — threshold: **{thresh}**  |  EMA: **{ema_w}**  |  Hold: **{min_hold} day(s)**")

    # Run backtest on test set
    y_proba_test  = xgb_model.predict_proba(X_test)
    final_signals = generate_signals(y_proba_test, X_test.index, thresh, f"EMA_{ema_w}", cross_data)
    bt            = backtest_with_hold(final_signals, X_test.index, min_hold, cross_data)

    total_mkt    = (bt["Market_Value"].iloc[-1]   / 1000 - 1) * 100
    total_strat  = (bt["Strategy_Value"].iloc[-1] / 1000 - 1) * 100
    sh_strat     = sharpe_ratio(bt["Strategy_Return"].dropna())
    sh_mkt       = sharpe_ratio(bt["Market_Return"].dropna())
    dd_strat     = max_drawdown(bt["Strategy_Value"])
    dd_mkt       = max_drawdown(bt["Market_Value"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Return — Strategy", f"{total_strat:.2f}%",
                delta=f"{total_strat - total_mkt:.2f}% vs market")
    col2.metric("Sharpe Ratio — Strategy", f"{sh_strat:.3f}",
                delta=f"{sh_strat - sh_mkt:.3f} vs market")
    col3.metric("Max Drawdown — Strategy", f"{dd_strat:.2f}%",
                delta=f"{dd_strat - dd_mkt:.2f}% vs market", delta_color="inverse")

    # 3-panel backtest chart
    buy_pts  = bt[bt["Prediction"] == 1]
    cash_pts = bt[bt["Prediction"] == 0]

    fig_bt = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.35, 0.20],
        subplot_titles=["Portfolio Value ($)", "Price + BUY/CASH Signals", "Daily Strategy Return"],
        vertical_spacing=0.06,
    )

    fig_bt.add_trace(go.Scatter(x=bt.index, y=bt["Market_Value"],
                                name="Market (B&H)", line=dict(color="steelblue")), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=bt.index, y=bt["Strategy_Value"],
                                name="Strategy (ML)", line=dict(color="darkorange")), row=1, col=1)

    fig_bt.add_trace(go.Scatter(x=bt.index, y=bt["Close"], name="Close",
                                line=dict(color="grey", width=0.8), showlegend=False), row=2, col=1)
    fig_bt.add_trace(go.Scatter(x=buy_pts.index, y=buy_pts["Close"], mode="markers",
                                name="BUY", marker=dict(color="green", size=5)), row=2, col=1)
    fig_bt.add_trace(go.Scatter(x=cash_pts.index, y=cash_pts["Close"], mode="markers",
                                name="CASH", marker=dict(color="red", size=4, opacity=0.4)),
                     row=2, col=1)

    ret_colors = ["forestgreen" if v >= 0 else "tomato"
                  for v in bt["Strategy_Return"].fillna(0)]
    fig_bt.add_trace(go.Bar(x=bt.index, y=bt["Strategy_Return"],
                            marker_color=ret_colors, name="Daily Return",
                            showlegend=False), row=3, col=1)
    fig_bt.add_hline(y=0, line_color="black", line_width=0.6, row=3, col=1)

    fig_bt.update_layout(height=750, legend=dict(orientation="h", y=1.04),
                         margin=dict(t=60, b=20))
    st.plotly_chart(fig_bt, use_container_width=True)

    # ── Position sizing comparison ─────────────────────────────────────────
    st.write("---")
    st.write("**Position Sizing vs Binary Strategy**")
    st.caption("Position sizing uses P(up) as the capital fraction (e.g. 70% confidence → 70% invested) "
               "instead of 100% BUY or 100% CASH.")

    pos_sizes = generate_position_sizes(
        xgb_model.predict_proba(X_test), X_test.index, f"EMA_{ema_w}", cross_data
    )
    bt_ps     = backtest_continuous(pos_sizes, X_test.index, cross_data)
    total_ps  = (bt_ps["Strategy_Value"].iloc[-1] / 1000 - 1) * 100
    sh_ps     = sharpe_ratio(bt_ps["Strategy_Return"].dropna())
    dd_ps     = max_drawdown(bt_ps["Strategy_Value"])

    cmp_df = pd.DataFrame({
        "Total Return (%)": [round(total_ps, 2), round(total_strat, 2), round(total_mkt, 2)],
        "Sharpe Ratio":     [round(sh_ps, 3),    round(sh_strat, 3),   round(sh_mkt, 3)],
        "Max Drawdown (%)": [round(dd_ps, 2),     round(dd_strat, 2),   round(dd_mkt, 2)],
    }, index=["Position Sizing", "Binary", "Market"])
    st.dataframe(cmp_df.style.highlight_max(color="lightgreen", axis=0)
                             .highlight_min(subset=["Max Drawdown (%)"], color="lightyellow", axis=0),
                 use_container_width=True)

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(x=bt.index, y=bt["Market_Value"],
                                 name="Market (B&H)", line=dict(color="steelblue")))
    fig_cmp.add_trace(go.Scatter(x=bt.index, y=bt["Strategy_Value"],
                                 name="Binary", line=dict(color="darkorange")))
    fig_cmp.add_trace(go.Scatter(x=bt_ps.index, y=bt_ps["Strategy_Value"],
                                 name="Position Sizing", line=dict(color="forestgreen", dash="dash")))
    fig_cmp.update_layout(height=380, legend=dict(orientation="h", y=1.04),
                          yaxis_title="Portfolio Value ($)", margin=dict(t=40, b=20))
    st.plotly_chart(fig_cmp, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# Tab 4 — Live Signal
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader(f"Today's Signal — {ticker}")

    latest       = cross_data.iloc[-1]
    latest_proba = xgb_model.predict_proba(cross_data[cross_features].iloc[[-1]])[0]

    p_up   = latest_proba[1]
    ema_ok = bool(latest["Close"] > latest[f"EMA_{ema_w}"])
    signal = "BUY" if (p_up > thresh and ema_ok) else "CASH"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Signal", signal)
    col2.metric("P(UP)", f"{p_up:.1%}")
    col3.metric("Close", f"${latest['Close']:.2f}")
    col4.metric(f"EMA {ema_w}", f"${latest[f'EMA_{ema_w}']:.2f}",
                delta="above EMA ✓" if ema_ok else "below EMA ✗",
                delta_color="normal" if ema_ok else "inverse")

    st.caption(f"Last trading day in dataset: **{cross_data.index[-1].date()}**")

    st.write("**P(UP) from all models**")
    proba_rows = []
    for name, model in models.items():
        p = model.predict_proba(cross_data[cross_features].iloc[[-1]])[0][1]
        proba_rows.append({
            "Model":  name,
            "P(UP)":  round(p, 4),
            "Signal": "BUY" if p > thresh else "CASH",
        })
    st.dataframe(pd.DataFrame(proba_rows).set_index("Model"), use_container_width=True)

    st.write("**Top-15 cross-asset feature values (last trading day)**")
    feat_df = (cross_data[cross_features].iloc[[-1]]
               .T.rename(columns={cross_data.index[-1]: "Value"})
               .round(4)
               .head(15))
    st.dataframe(feat_df, use_container_width=True)
