import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Portfolio Intelligence Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS
# =============================================================================
st.markdown(
    """
<style>
    .main { background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%); }
    .stMetric { background: rgba(30, 41, 59, 0.4); padding: 1rem; border-radius: 0.5rem; border: 2px solid rgba(71, 85, 105, 0.5); }
    .summary-box { background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.1));
                   border: 2px solid rgba(59, 130, 246, 0.3); border-radius: 1rem; padding: 1.5rem; margin-bottom: 1rem; }
    .warn-box { background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 0.75rem; padding: 0.9rem; margin: 0.5rem 0 1rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# HEADER
# =============================================================================
st.title("🎯 Portfolio Intelligence Dashboard")
st.markdown(f"**Analysis Date:** {datetime.now().strftime('%B %d, %Y')}")

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    ALL_TICKERS = [
        "SPY", "GLD", "SMH", "COPX", "TLT", "FXI", "UUP", "USO",
        "QQQ", "IWM", "EEM", "VNQ", "DBC", "GDX", "SLV"
    ]
    default_tickers = ["SPY", "GLD", "SMH", "COPX", "TLT", "FXI", "UUP", "USO"]

    tickers = st.multiselect(
        "Select ETFs",
        options=ALL_TICKERS,
        default=default_tickers,
        help="SPY is required as the market benchmark",
    )

    window = st.slider(
        "Rolling Window (days)",
        min_value=10,
        max_value=60,
        value=30,
        help="Window for rolling correlation calculation",
    )

    period = st.selectbox(
        "Analysis Period",
        ["3mo", "6mo", "1y", "2y"],
        index=1,
        help="Longer period provides more reliable trend signals",
    )

    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("**ETF Guide:**")
    st.caption("SPY = S&P 500 | GLD = Gold | SMH = Semis | COPX = Copper")
    st.caption("TLT = 20Y+ Treasuries | FXI = China LC | UUP = USD | USO = Oil")


# =============================================================================
# HELPERS
# =============================================================================
def _normalize_tickers(tickers_list):
    # Uppercase, strip, unique while preserving order
    seen = set()
    out = []
    for t in tickers_list:
        tt = str(t).strip().upper()
        if tt and tt not in seen:
            seen.add(tt)
            out.append(tt)
    return out


def _extract_prices(download_df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance returns different shapes depending on tickers count / settings.
    This extracts an "Adj Close" (preferred) or "Close" table with columns=tickers.
    """
    if download_df is None or download_df.empty:
        return pd.DataFrame()

    cols = download_df.columns

    # Case 1: MultiIndex columns: (Field, Ticker)
    if isinstance(cols, pd.MultiIndex):
        level0 = cols.get_level_values(0)
        if "Adj Close" in level0:
            prices = download_df.xs("Adj Close", axis=1, level=0, drop_level=True)
        elif "Close" in level0:
            prices = download_df.xs("Close", axis=1, level=0, drop_level=True)
        else:
            # fallback: pick first field
            first_field = cols.levels[0][0]
            prices = download_df.xs(first_field, axis=1, level=0, drop_level=True)
        prices.columns = [str(c).upper() for c in prices.columns]
        return prices

    # Case 2: Single-level columns; sometimes already tickers, sometimes fields
    # If it contains "Adj Close" as a column, it's a single-ticker field table.
    if "Adj Close" in cols:
        return download_df[["Adj Close"]].rename(columns={"Adj Close": "SINGLE"})
    if "Close" in cols:
        return download_df[["Close"]].rename(columns={"Close": "SINGLE"})

    # Otherwise assume columns are tickers
    prices = download_df.copy()
    prices.columns = [str(c).upper() for c in prices.columns]
    return prices


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers_list, period_str):
    """
    Fetch prices and return (prices_df, missing_tickers, available_tickers)
    Robust to partial failures (some tickers missing).
    """
    tks = _normalize_tickers(tickers_list)
    if not tks:
        return pd.DataFrame(), tks, []

    # Download once. yfinance can partially succeed; we'll detect missing columns.
    try:
        raw = yf.download(
            tickers=tks,
            period=period_str,
            progress=False,
            auto_adjust=False,
            threads=True,
            group_by="column",
        )
    except Exception:
        # One retry (helps transient Yahoo hiccups)
        raw = yf.download(
            tickers=tks,
            period=period_str,
            progress=False,
            auto_adjust=False,
            threads=True,
            group_by="column",
        )

    prices = _extract_prices(raw)

    # If SINGLE (happens if yfinance returns single-ticker fields), try to map back
    if list(prices.columns) == ["SINGLE"] and len(tks) == 1:
        prices = prices.rename(columns={"SINGLE": tks[0]})

    # Clean: drop empty columns
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")

    available = [c for c in prices.columns if c in set(tks)]
    missing = [t for t in tks if t not in set(prices.columns)]

    # Keep only requested tickers (and in request order)
    prices = prices[[c for c in tks if c in prices.columns]]

    return prices, missing, available


def calculate_correlation(x, y) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    c = np.corrcoef(x, y)[0, 1]
    return float(c) if np.isfinite(c) else np.nan


def calculate_beta(asset_returns, market_returns) -> float:
    if len(asset_returns) < 2 or len(market_returns) < 2:
        return np.nan
    cov = np.cov(asset_returns, market_returns)[0, 1]
    var = np.var(market_returns)
    if not np.isfinite(cov) or not np.isfinite(var) or var <= 0:
        return np.nan
    return float(cov / var)


def calculate_dual_beta(asset_returns, market_returns):
    up_mask = market_returns > 0
    down_mask = market_returns < 0

    beta_up = calculate_beta(asset_returns[up_mask], market_returns[up_mask]) if up_mask.sum() > 1 else np.nan
    beta_down = calculate_beta(asset_returns[down_mask], market_returns[down_mask]) if down_mask.sum() > 1 else np.nan
    return beta_up, beta_down


def calculate_trend_signal(prices: pd.Series):
    """
    50/200 MA regime:
      Positive ↑ : price > MA50 and > MA200
      Negative ↓ : price < MA50 and < MA200
      No Signal  : otherwise or insufficient history
    """
    prices = prices.dropna()
    if len(prices) < 210:
        return "No Signal", 0

    ma_50 = prices.rolling(50).mean()
    ma_200 = prices.rolling(200).mean()

    p = prices.iloc[-1]
    m50 = ma_50.iloc[-1]
    m200 = ma_200.iloc[-1]

    if p > m50 and p > m200:
        return "Positive ↑", 1
    if p < m50 and p < m200:
        return "Negative ↓", -1
    return "No Signal", 0


# =============================================================================
# MAIN
# =============================================================================
tickers = _normalize_tickers(tickers)

if len(tickers) < 2:
    st.warning("⚠️ Please select at least 2 tickers (including SPY).")
    st.stop()

if "SPY" not in tickers:
    st.warning("⚠️ SPY is required as the market benchmark.")
    st.stop()

with st.spinner("📊 Fetching data and running analysis..."):
    prices, missing, available = fetch_prices(tickers, period)

if prices.empty or len(available) < 2:
    st.error("❌ Failed to fetch enough data from Yahoo Finance.")
    if missing:
        st.info(f"Missing/failed tickers: {', '.join(missing)}")
    st.info("💡 Try fewer tickers, a shorter period, or refresh.")
    st.stop()

# Show partial failure info (THIS is what prevents KeyError: 'SPY' / 'GLD')
if missing:
    st.markdown(
        f"""
<div class="warn-box">
<b>Some tickers could not be downloaded</b><br>
Missing: <code>{', '.join(missing)}</code><br>
Continuing analysis with: <code>{', '.join(available)}</code>
</div>
""",
        unsafe_allow_html=True,
    )

# Enforce benchmark exists in downloaded data
if "SPY" not in prices.columns:
    st.error("❌ SPY data could not be downloaded from Yahoo Finance, so the benchmark is unavailable.")
    st.info("💡 Refresh, change period, or try again later (Yahoo can intermittently fail).")
    st.stop()

# Use only the tickers that actually exist
tickers_live = [t for t in tickers if t in prices.columns]

# Returns
returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")

# Drop columns that are all NaN in returns (can happen if one ticker has 1 datapoint)
returns = returns.dropna(axis=1, how="all")
tickers_live = [t for t in tickers_live if t in returns.columns]

if "SPY" not in tickers_live or len(tickers_live) < 2:
    st.error("❌ Not enough clean return series to run correlation/beta analysis.")
    st.stop()

# Rolling correlation vs SPY
rolling_corr = {}
for t in tickers_live:
    if t == "SPY":
        continue
    # Align series to avoid KeyErrors and silent NaN issues
    a = returns[t].dropna()
    m = returns["SPY"].dropna()
    aligned = pd.concat([a, m], axis=1, join="inner").dropna()
    if aligned.shape[0] < window + 2:
        rolling_corr[t] = pd.Series(index=returns.index, dtype=float)
    else:
        rolling_corr[t] = aligned.iloc[:, 0].rolling(window=window).corr(aligned.iloc[:, 1])

rolling_corr_df = pd.DataFrame(rolling_corr)

# Recent window
recent_returns = returns.tail(window)

# Metrics
metrics = []
for t in tickers_live:
    if t == "SPY":
        continue

    aligned = pd.concat([recent_returns[t], recent_returns["SPY"]], axis=1, join="inner").dropna()
    if aligned.shape[0] < 5:
        # Not enough data
        corr = np.nan
        beta_up, beta_down = np.nan, np.nan
        vol = np.nan
        cum_return = np.nan
    else:
        corr = calculate_correlation(aligned.iloc[:, 0].values, aligned.iloc[:, 1].values)
        beta_up, beta_down = calculate_dual_beta(aligned.iloc[:, 0].values, aligned.iloc[:, 1].values)
        vol = aligned.iloc[:, 0].std() * np.sqrt(252) * 100
        cum_return = (1 + aligned.iloc[:, 0]).prod() - 1

    asymmetry = (beta_up - beta_down) if np.isfinite(beta_up) and np.isfinite(beta_down) else np.nan
    trend_signal, trend_value = calculate_trend_signal(prices[t])

    # Correlation trend (last 60 observations of rolling corr if available)
    corr_change = 0.0
    corr_trend = "→"
    if t in rolling_corr_df.columns:
        s = rolling_corr_df[t].dropna()
        if len(s) > 60:
            corr_change = float(s.iloc[-1] - s.iloc[-60])
            corr_trend = "↓" if corr_change < -0.05 else "↑" if corr_change > 0.05 else "→"

    metrics.append(
        {
            "Ticker": t,
            "Correlation": corr,
            "Corr_Change": corr_change,
            "Corr_Trend": corr_trend,
            "Volatility": float(vol) if np.isfinite(vol) else np.nan,
            "Up Beta": float(beta_up) if np.isfinite(beta_up) else np.nan,
            "Down Beta": float(beta_down) if np.isfinite(beta_down) else np.nan,
            "Asymmetry": float(asymmetry) if np.isfinite(asymmetry) else np.nan,
            "30D Return": float(cum_return * 100) if np.isfinite(cum_return) else np.nan,
            "Trend": trend_signal,
            "Trend_Value": trend_value,
        }
    )

metrics_df = pd.DataFrame(metrics)

# Correlation matrix (for downloaded tickers)
corr_matrix = returns[tickers_live].corr()

# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================
avg_corr = float(np.nanmean(metrics_df["Correlation"].values)) if not metrics_df.empty else np.nan

trend_diversifiers = metrics_df[
    (metrics_df["Trend_Value"] == 1) & (metrics_df["Correlation"].abs() < 0.5)
]["Ticker"].tolist()

avoid_assets = metrics_df[(metrics_df["Correlation"] > 0.7)]["Ticker"].tolist()

good_hedges = metrics_df[
    (metrics_df["Correlation"] < -0.3) & (metrics_df["Asymmetry"] > 0.15)
].sort_values("Asymmetry", ascending=False)["Ticker"].tolist()

high_corr_trends = metrics_df[
    (metrics_df["Trend_Value"] == 1) & (metrics_df["Correlation"] > 0.7)
]["Ticker"].tolist()

negative_trends = metrics_df[(metrics_df["Trend_Value"] == -1)]["Ticker"].tolist()

st.markdown(
    """
<div class="summary-box">
  <h3 style="margin:0; color:#60a5fa;">📊 Executive Summary</h3>
</div>
""",
    unsafe_allow_html=True,
)

summary_text = f"""
**Best Opportunities (Trend + Diversification):** {
    f"**{', '.join(trend_diversifiers)}** show positive technical momentum (above 50D/200D MA) with correlation below 0.5."
    if trend_diversifiers else
    "No assets currently meet both trend + diversification criteria."
}

**Recommended Hedges:** {
    f"**{', '.join(good_hedges[:3])}** exhibit negative correlation and convex payoff (asymmetry > 0.15)."
    if good_hedges else
    "Limited hedge candidates in the selected set."
}

**Assets to Avoid/Reduce:** {
    f"**{', '.join(avoid_assets)}** show correlation > 0.7 with SPY (concentrated equity beta)."
    if avoid_assets else
    "No highly correlated assets identified."
}{
    f" Additionally, **{', '.join(high_corr_trends)}** have positive trends but high correlation—tactical only."
    if high_corr_trends else ""
}{
    f" **{', '.join(negative_trends)}** are in negative trends—avoid until improvement."
    if negative_trends else ""
}

**Market Regime:** {
    f"Elevated correlations (avg {avg_corr:.2f}). Focus on hedges / tighter risk controls."
    if np.isfinite(avg_corr) and avg_corr > 0.5 else
    f"Balanced regime (avg {avg_corr:.2f}). Favor selective diversifiers + momentum."
    if np.isfinite(avg_corr) and avg_corr > 0.2 else
    f"Lower-correlation regime (avg {avg_corr:.2f}). Diversification is working."
    if np.isfinite(avg_corr) else
    "Insufficient data to classify regime."
}
"""
st.markdown(summary_text)

# =============================================================================
# LAYOUT
# =============================================================================
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.subheader("Correlation Matrix")
    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale=[
                [0, "#10b981"],
                [0.5, "#1e293b"],
                [1, "#ef4444"],
            ],
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text:.2f}",
            textfont={"size": 10, "color": "white"},
            colorbar=dict(
                title="Correlation",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"],
            ),
            hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>",
        )
    )
    fig_heatmap.update_layout(template="plotly_dark", height=420, xaxis_title="", yaxis_title="", font=dict(size=10))
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.markdown("**Color Guide:** 🟢 Negative = hedge-like | ⚫ ~0 = diversifier | 🔴 Positive = correlated")

with col2:
    st.subheader(f"Rolling {window}D Correlation vs SPY")
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
    fig_corr.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, line_width=1)
    fig_corr.add_hline(y=0.8, line_dash="dot", line_color="red", opacity=0.3, annotation_text="High Corr (0.8)", annotation_position="right")
    fig_corr.add_hline(y=-0.3, line_dash="dot", line_color="green", opacity=0.3, annotation_text="Hedge Zone (-0.3)", annotation_position="right")
    fig_corr.update_layout(
        template="plotly_dark",
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        xaxis_title="",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1, 1]),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with col3:
    st.subheader("Market Regime")
    high_corr_count = int((metrics_df["Correlation"].abs() > 0.7).sum()) if not metrics_df.empty else 0
    hedge_count = int((metrics_df["Correlation"] < -0.3).sum()) if not metrics_df.empty else 0
    positive_trend_count = int((metrics_df["Trend_Value"] == 1).sum()) if not metrics_df.empty else 0

    st.metric("Avg Correlation", f"{avg_corr:.2f}" if np.isfinite(avg_corr) else "NA")
    st.metric("High Corr Assets", f"{high_corr_count}/{len(metrics_df)}" if len(metrics_df) else "0/0")
    st.metric("Hedge Candidates", f"{hedge_count}")
    st.metric("Positive Trends", f"{positive_trend_count}/{len(metrics_df)}" if len(metrics_df) else "0/0")

    if np.isfinite(avg_corr) and avg_corr > 0.5:
        st.error("⚠️ **Risk-On**\nHigh correlation regime")
    elif np.isfinite(avg_corr) and avg_corr < 0.2:
        st.success("✅ **Diversified**\nLow correlation regime")
    else:
        st.info("ℹ️ **Mixed**\nModerate correlation")

# =============================================================================
# BETA ASYMMETRY
# =============================================================================
st.subheader("Beta Asymmetry Analysis")

col_beta1, col_beta2 = st.columns([3, 1])

with col_beta1:
    fig_beta = go.Figure()
    fig_beta.add_trace(
        go.Bar(
            name="Up Beta (Greed)",
            x=metrics_df["Ticker"],
            y=metrics_df["Up Beta"],
            text=metrics_df["Up Beta"].round(2),
            textposition="outside",
        )
    )
    fig_beta.add_trace(
        go.Bar(
            name="Down Beta (Fear)",
            x=metrics_df["Ticker"],
            y=metrics_df["Down Beta"],
            text=metrics_df["Down Beta"].round(2),
            textposition="outside",
        )
    )
    fig_beta.update_layout(
        template="plotly_dark",
        height=320,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="",
        yaxis_title="Beta",
        showlegend=True,
    )
    st.plotly_chart(fig_beta, use_container_width=True)

with col_beta2:
    st.markdown("**Asymmetry Guide:**")
    st.markdown("**> 0.2** = Convex (Good hedge)")
    st.markdown("**0 to 0.2** = Neutral")
    st.markdown("**< 0** = Concave (Risk)")

    good_asym = metrics_df[metrics_df["Asymmetry"] > 0.2].sort_values("Asymmetry", ascending=False)
    if len(good_asym):
        st.markdown("**Best Asymmetry:**")
        for _, r in good_asym.head(5).iterrows():
            st.markdown(f"✅ **{r['Ticker']}**: {r['Asymmetry']:.2f}")
    else:
        st.markdown("*No assets with asymmetry > 0.2*")

# =============================================================================
# TABLE
# =============================================================================
st.subheader("Detailed Metrics")

display_df = metrics_df.copy()

def _fmt(x, f, suffix=""):
    return "NA" if pd.isna(x) else (f.format(x) + suffix)

display_df["Correlation"] = display_df["Correlation"].apply(lambda x: _fmt(x, "{:.2f}"))
display_df["Volatility"] = display_df["Volatility"].apply(lambda x: _fmt(x, "{:.1f}", "%"))
display_df["Up Beta"] = display_df["Up Beta"].apply(lambda x: _fmt(x, "{:.2f}"))
display_df["Down Beta"] = display_df["Down Beta"].apply(lambda x: _fmt(x, "{:.2f}"))
display_df["Asymmetry"] = display_df["Asymmetry"].apply(lambda x: _fmt(x, "{:.2f}"))
display_df["30D Return"] = display_df["30D Return"].apply(lambda x: _fmt(x, "{:.1f}", "%"))

def get_status(row):
    try:
        corr = float(row["Correlation"])
    except Exception:
        return "⚪ NA"
    if corr < -0.3:
        return "🟢 Hedge"
    if abs(corr) < 0.3:
        return "🟡 Diversifier"
    if corr > 0.7:
        return "🔴 Concentrated"
    return "🔵 Growth"

display_df["Status"] = display_df.apply(get_status, axis=1)

st.dataframe(
    display_df[
        [
            "Ticker",
            "Status",
            "Correlation",
            "Corr_Trend",
            "Trend",
            "Volatility",
            "Up Beta",
            "Down Beta",
            "Asymmetry",
            "30D Return",
        ]
    ],
    use_container_width=True,
    height=360,
)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption(
    f"📊 **Analysis Period:** {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')} | "
    f"**Rolling Window:** {window} days | **Tickers Used:** {', '.join(tickers_live)} | "
    f"**Trend:** 50D/200D MA | **Source:** Yahoo Finance | **Cache:** 1 hour"
)
