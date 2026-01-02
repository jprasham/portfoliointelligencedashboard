import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Portfolio Intelligence Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Custom CSS
# =========================
st.markdown(
    """
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    .stMetric {
        background: rgba(30, 41, 59, 0.4);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid rgba(71, 85, 105, 0.5);
    }
    .insight-card {
        padding: 1.25rem;
        border-radius: 1rem;
        border: 2px solid;
        margin-bottom: 0.9rem;
        background: rgba(30, 41, 59, 0.25);
    }
    .insight-high {
        border-color: rgba(16, 185, 129, 0.55);
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(16, 185, 129, 0.04));
    }
    .insight-medium {
        border-color: rgba(245, 158, 11, 0.55);
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(245, 158, 11, 0.04));
    }
    .insight-low {
        border-color: rgba(148, 163, 184, 0.40);
        background: linear-gradient(135deg, rgba(148, 163, 184, 0.10), rgba(148, 163, 184, 0.03));
    }
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Title
# =========================
st.title("🎯 Portfolio Intelligence Dashboard")
st.markdown("### Rolling Correlation & Beta Analysis (SPY benchmark)")

# =========================
# Sidebar configuration
# =========================
with st.sidebar:
    st.header("Configuration")

    default_tickers = ["SPY", "GLD", "SMH", "COPX", "TLT", "FXI", "UUP"]
    options = ["SPY", "GLD", "SMH", "COPX", "TLT", "FXI", "UUP", "QQQ", "IWM", "EEM", "VNQ", "DBC"]

    tickers = st.multiselect(
        "Select ETFs to analyze",
        options=options,
        default=default_tickers,
        help="SPY will be used as the market benchmark",
    )

    window = st.slider("Rolling Window (days)", 10, 60, 30)
    period = st.selectbox("Analysis Period", ["3mo", "6mo", "1y", "2y"], index=1)

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()

# =========================
# Helpers
# =========================
def _extract_adj_close(raw: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance can return:
      - single-level columns (e.g., 'Adj Close' only)
      - MultiIndex columns with levels like ('Adj Close', 'GLD') or ('GLD','Adj Close')
    This function extracts an Adj Close DataFrame with columns=tickers.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    cols = raw.columns
    if isinstance(cols, pd.MultiIndex):
        # Case A: level 0 contains 'Adj Close'
        if "Adj Close" in cols.get_level_values(0):
            adj = raw["Adj Close"].copy()
        # Case B: level 1 contains 'Adj Close'
        elif "Adj Close" in cols.get_level_values(1):
            adj = raw.xs("Adj Close", level=1, axis=1).copy()
        else:
            return pd.DataFrame()

        # Ensure columns are plain tickers (strings)
        adj.columns = [str(c) for c in adj.columns]
        return adj

    # Single index columns
    # If we downloaded 1 ticker, yfinance might give a Series or DF with OHLCV cols
    if "Adj Close" in cols:
        adj = raw[["Adj Close"]].copy()
        # Not enough info to name the ticker from raw in this branch; handled in fetch_data
        return adj

    return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_data(requested_tickers: list[str], period: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Fetch adjusted close prices for requested_tickers.
    Returns: (adj_close_df, available_tickers, missing_tickers)
    """
    requested_tickers = [t.strip().upper() for t in requested_tickers if str(t).strip()]
    requested_tickers = list(dict.fromkeys(requested_tickers))  # de-dupe, preserve order

    if not requested_tickers:
        return pd.DataFrame(), [], []

    try:
        raw = yf.download(
            tickers=requested_tickers,
            period=period,
            progress=False,
            group_by="column",
            auto_adjust=False,
            threads=True,
        )

        # Extract Adj Close robustly
        adj = _extract_adj_close(raw)

        # If we got single-level 'Adj Close' column only (often when 1 ticker),
        # yfinance returns a DF with one column 'Adj Close'. Rename it to the ticker.
        if not adj.empty and list(adj.columns) == ["Adj Close"] and len(requested_tickers) == 1:
            adj = adj.rename(columns={"Adj Close": requested_tickers[0]})

        # Sometimes yfinance returns a Series if only 1 ticker and slicing happened earlier
        if isinstance(adj, pd.Series):
            adj = adj.to_frame(name=requested_tickers[0] if requested_tickers else "PRICE")

        # Clean
        adj = adj.dropna(how="all")
        if not adj.empty:
            adj = adj.sort_index()

        available = [t for t in requested_tickers if t in adj.columns]
        missing = [t for t in requested_tickers if t not in adj.columns]

        # Keep columns in requested order
        adj = adj[available] if available else pd.DataFrame(index=adj.index)

        return adj, available, missing

    except Exception:
        # Avoid leaking exception details on Streamlit Cloud logs; show a friendly message
        return pd.DataFrame(), [], requested_tickers


def calculate_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def calculate_beta(asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
    if len(asset_returns) < 2 or len(market_returns) < 2:
        return 0.0
    cov = np.cov(asset_returns, market_returns)[0, 1]
    var = np.var(market_returns)
    return float(cov / var) if var > 0 else 0.0


def calculate_dual_beta(asset_returns: np.ndarray, market_returns: np.ndarray) -> tuple[float, float]:
    up_mask = market_returns > 0
    down_mask = market_returns < 0

    beta_up = calculate_beta(asset_returns[up_mask], market_returns[up_mask]) if up_mask.sum() > 1 else 0.0
    beta_down = calculate_beta(asset_returns[down_mask], market_returns[down_mask]) if down_mask.sum() > 1 else 0.0
    return beta_up, beta_down


# =========================
# Main
# =========================
if len(tickers) < 2:
    st.warning("Please select at least 2 tickers (including SPY).")
    st.stop()

if "SPY" not in tickers:
    st.warning("SPY must be included as the market benchmark.")
    st.stop()

with st.spinner("Fetching data and running analysis..."):
    data, available_tickers, missing_tickers = fetch_data(tickers, period)

if missing_tickers:
    st.warning(
        f"Some tickers returned no data from Yahoo Finance and were skipped: {', '.join(missing_tickers)}"
    )

if data is None or data.empty:
    st.error("Failed to fetch usable price data. Try a different period or fewer tickers.")
    st.stop()

if "SPY" not in data.columns:
    st.error("SPY data could not be fetched, so the benchmark is unavailable. Please refresh or try again.")
    st.stop()

# Need at least one non-SPY asset available
non_spy = [t for t in available_tickers if t != "SPY"]
if not non_spy:
    st.error("No non-SPY assets returned data. Please select other tickers.")
    st.stop()

# Returns
returns = data.pct_change().dropna()
if returns.empty or len(returns) < max(3, window):
    st.error("Not enough return observations for the selected period/window. Try a longer period.")
    st.stop()

# Rolling correlation vs SPY
rolling_corr = {
    t: returns[t].rolling(window=window).corr(returns["SPY"])
    for t in non_spy
}
rolling_corr_df = pd.DataFrame(rolling_corr)

# Recent window
recent_returns = returns.tail(window)

# Metrics
metrics = []
for t in non_spy:
    x = recent_returns[t].values
    m = recent_returns["SPY"].values

    corr = calculate_correlation(x, m)
    beta_up, beta_down = calculate_dual_beta(x, m)
    asymmetry = beta_up - beta_down

    vol = recent_returns[t].std() * np.sqrt(252) * 100.0
    cum_return = (1.0 + recent_returns[t]).prod() - 1.0

    # correlation trend vs 60 trading days if available
    if len(rolling_corr_df) > 60:
        recent_corr = float(rolling_corr_df[t].iloc[-1])
        earlier_corr = float(rolling_corr_df[t].iloc[-60])
        trend = "Falling" if recent_corr < earlier_corr else "Rising"
    else:
        trend = "Stable"

    if abs(corr) < 0.3:
        div_score = "High"
    elif abs(corr) < 0.7:
        div_score = "Medium"
    else:
        div_score = "Low"

    hedge_quality = "Good" if asymmetry > 0.2 else "Poor"

    metrics.append(
        {
            "Ticker": t,
            "Correlation": corr,
            "Volatility": vol,
            "Up Beta": beta_up,
            "Down Beta": beta_down,
            "Asymmetry": asymmetry,
            "Trend": trend,
            "Diversification": div_score,
            "Hedge Quality": hedge_quality,
            "30D Return": cum_return * 100.0,
            "Up Days": int((recent_returns["SPY"] > 0).sum()),
            "Down Days": int((recent_returns["SPY"] < 0).sum()),
        }
    )

metrics_df = pd.DataFrame(metrics)
if metrics_df.empty:
    st.error("No metrics to display (likely due to missing data).")
    st.stop()

# =========================
# Insights
# =========================
insights = []

good_hedges = metrics_df.loc[metrics_df["Hedge Quality"] == "Good", "Ticker"].tolist()
if good_hedges:
    insights.append(
        {
            "type": "high",
            "title": "🛡️ Best Hedges Right Now",
            "description": f"{', '.join(good_hedges)} show asymmetric protection (participate more in gains than losses).",
            "action": "Consider 10–20% allocation for downside protection (depending on risk).",
        }
    )

for _, row in metrics_df.iterrows():
    if row["Trend"] == "Falling" and row["Correlation"] < 0:
        insights.append(
            {
                "type": "high",
                "title": f"📉 {row['Ticker']} Correlation Shifting Negative",
                "description": f"{row['Ticker']} correlation with SPY is decreasing and is now {row['Correlation']:.2f}.",
                "action": "Diversification benefit appears to be improving; reassess sizing.",
            }
        )

high_div = metrics_df.loc[metrics_df["Diversification"] == "High", "Ticker"].tolist()
if high_div:
    insights.append(
        {
            "type": "high",
            "title": "🎯 Strong Diversifiers",
            "description": f"{', '.join(high_div)} have low correlation (|corr| < 0.3).",
            "action": "Useful for reducing portfolio volatility via diversification.",
        }
    )

high_corr = metrics_df.loc[metrics_df["Correlation"].abs() > 0.8, "Ticker"].tolist()
if high_corr:
    insights.append(
        {
            "type": "medium",
            "title": "⚠️ Concentration Warning",
            "description": f"{', '.join(high_corr)} are highly correlated with SPY (limited diversification).",
            "action": "Consider lowering exposure or adding lower-correlation alternatives.",
        }
    )

top_performer = metrics_df.loc[metrics_df["30D Return"].idxmax()]
insights.append(
    {
        "type": "low",
        "title": "📈 Top Performer",
        "description": f"{top_performer['Ticker']} returned {top_performer['30D Return']:.1f}% over the last {window} days.",
        "action": "Monitor momentum; watch for mean reversion.",
    }
)

# Display insights
st.header("💡 Actionable Market Signals")
for ins in insights:
    st.markdown(
        f"""
<div class="insight-card insight-{ins['type']}">
  <h3 style="margin:0 0 0.25rem 0;">{ins['title']}</h3>
  <p style="margin:0.25rem 0 0.6rem 0;">{ins['description']}</p>
  <p style="margin:0; font-style: italic; color:#94a3b8;">→ {ins['action']}</p>
</div>
""",
        unsafe_allow_html=True,
    )

# =========================
# Asset cards
# =========================
st.header("📊 Asset Overview")
cols = st.columns(min(len(metrics_df), 6))
for idx, (_, asset) in enumerate(metrics_df.iterrows()):
    with cols[idx % len(cols)]:
        c = float(asset["Correlation"])
        if abs(c) < 0.3:
            corr_color = "🟢"
        elif abs(c) < 0.7:
            corr_color = "🟡"
        else:
            corr_color = "🔴"

        st.metric(
            label=f"{asset['Ticker']} {corr_color}",
            value=f"{c:.2f}",
            delta=f"{asset['30D Return']:.1f}%",
            help=f"Correlation: {c:.2f}\nVolatility: {asset['Volatility']:.1f}%",
        )
        st.caption(f"Vol: {asset['Volatility']:.1f}% | {asset['Diversification']} Div")

# =========================
# Charts
# =========================
st.header("📈 Market Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Rolling {window}-Day Correlation vs SPY")
    fig_corr = go.Figure()
    for t in rolling_corr_df.columns:
        fig_corr.add_trace(
            go.Scatter(
                x=rolling_corr_df.index,
                y=rolling_corr_df[t],
                mode="lines",
                name=t,
                line=dict(width=2.5),
            )
        )
    fig_corr.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_corr.update_layout(
        template="plotly_dark",
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1, 1]),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with col2:
    st.subheader("Beta Asymmetry")
    fig_beta = go.Figure()
    fig_beta.add_trace(go.Bar(name="Up Beta", x=metrics_df["Ticker"], y=metrics_df["Up Beta"]))
    fig_beta.add_trace(go.Bar(name="Down Beta", x=metrics_df["Ticker"], y=metrics_df["Down Beta"]))
    fig_beta.update_layout(
        template="plotly_dark",
        height=400,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Asset",
        yaxis_title="Beta",
    )
    st.plotly_chart(fig_beta, use_container_width=True)

# =========================
# Metrics table
# =========================
st.header("📋 Complete Portfolio Metrics")

display_df = metrics_df.copy()
display_df["Correlation"] = display_df["Correlation"].map("{:.2f}".format)
display_df["Volatility"] = display_df["Volatility"].map("{:.1f}%".format)
display_df["Up Beta"] = display_df["Up Beta"].map("{:.2f}".format)
display_df["Down Beta"] = display_df["Down Beta"].map("{:.2f}".format)
display_df["Asymmetry"] = display_df["Asymmetry"].map("{:.2f}".format)
display_df["30D Return"] = display_df["30D Return"].map("{:.1f}%".format)

display_df["Role"] = display_df.apply(
    lambda x: "🛡️ Hedge"
    if x["Hedge Quality"] == "Good"
    else ("🎯 Diversifier" if x["Diversification"] == "High" else "📈 Growth"),
    axis=1,
)

st.dataframe(
    display_df[
        ["Ticker", "Correlation", "Volatility", "Up Beta", "Down Beta", "Asymmetry", "Trend", "30D Return", "Role"]
    ],
    use_container_width=True,
    height=400,
)

# =========================
# Framework blocks
# =========================
st.header("🎯 Portfolio Construction Framework")

c1, c2 = st.columns(2)

with c1:
    st.success("**✅ Positive Signals**")
    st.markdown(
        """
- **Falling Correlations**: Assets decoupling = better diversification  
- **Negative Correlations**: True hedge candidates (zig when market zags)  
- **High Beta Asymmetry (>0.2)**: Convex payoff (wins bigger than losses)  
- **Stable Low Correlation**: Reliable diversifier  
"""
    )

with c2:
    st.error("**⚠️ Warning Signals**")
    st.markdown(
        """
- **Rising Correlations**: Crowded trades, contagion risk  
- **High Correlation (>0.8)**: Minimal diversification benefit  
- **Negative Asymmetry**: Concave payoff (losses bigger than gains)  
- **Sudden Correlation Spike**: Potential regime change  
"""
    )

st.info(
    """
**💡 Suggested Allocation Framework:**
- **Hedges (10–20%)**: Assets with negative down beta or high asymmetry  
- **Diversifiers (20–30%)**: Low correlation assets (|corr| < 0.3)  
- **Core/Satellite (50–70%)**: Market exposure balanced with defensive positions  
"""
)

# =========================
# Footer
# =========================
st.markdown("---")
st.caption(
    f"📊 Analysis Period: {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')} | "
    f"Rolling Window: {window} days | "
    f"Tickers (available): {', '.join(['SPY'] + non_spy)} | "
    f"Data Source: Yahoo Finance"
)
