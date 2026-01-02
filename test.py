import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Portfolio Intelligence Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
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
        padding: 1.5rem;
        border-radius: 1rem;
        border: 2px solid;
        margin-bottom: 1rem;
    }
    .insight-high {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05));
        border-color: rgba(16, 185, 129, 0.5);
    }
    .insight-medium {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.05));
        border-color: rgba(245, 158, 11, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🎯 Portfolio Intelligence Dashboard")
st.markdown("### Rolling 30-Day Correlation & Beta Analysis")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # Ticker selection
    default_tickers = ['SPY', 'GLD', 'SMH', 'COPX', 'TLT', 'FXI', 'UUP']
    tickers = st.multiselect(
        "Select ETFs to analyze",
        options=['SPY', 'GLD', 'SMH', 'COPX', 'TLT', 'FXI', 'UUP', 'QQQ', 'IWM', 'EEM', 'VNQ', 'DBC'],
        default=default_tickers,
        help="SPY will be used as the market benchmark"
    )
    
    # Rolling window
    window = st.slider("Rolling Window (days)", 10, 60, 30)
    
    # Date range
    period = st.selectbox("Analysis Period", ['3mo', '6mo', '1y', '2y'], index=1)
    
    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()

# Helper functions
@st.cache_data(ttl=3600)
def fetch_data(tickers, period):
    """Fetch historical price data"""
    try:
        data = yf.download(tickers, period=period, progress=False)['Adj Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def calculate_correlation(x, y):
    """Calculate Pearson correlation"""
    if len(x) < 2 or len(y) < 2:
        return 0
    return np.corrcoef(x, y)[0, 1]

def calculate_beta(asset_returns, market_returns):
    """Calculate beta"""
    if len(asset_returns) < 2 or len(market_returns) < 2:
        return 0
    covariance = np.cov(asset_returns, market_returns)[0, 1]
    variance = np.var(market_returns)
    return covariance / variance if variance > 0 else 0

def calculate_dual_beta(asset_returns, market_returns):
    """Calculate up and down beta"""
    up_mask = market_returns > 0
    down_mask = market_returns < 0
    
    beta_up = calculate_beta(
        asset_returns[up_mask], 
        market_returns[up_mask]
    ) if up_mask.sum() > 1 else 0
    
    beta_down = calculate_beta(
        asset_returns[down_mask], 
        market_returns[down_mask]
    ) if down_mask.sum() > 1 else 0
    
    return beta_up, beta_down

# Main analysis
if len(tickers) < 2:
    st.warning("Please select at least 2 tickers (including SPY)")
elif 'SPY' not in tickers:
    st.warning("SPY must be included as the market benchmark")
else:
    with st.spinner("Fetching data and running analysis..."):
        # Fetch data
        data = fetch_data(tickers, period)
        
        if data is not None and not data.empty:
            # Calculate returns
            returns = data.pct_change().dropna()
            
            # Rolling correlation
            rolling_corr = {}
            for ticker in tickers:
                if ticker != 'SPY':
                    rolling_corr[ticker] = returns[ticker].rolling(window=window).corr(returns['SPY'])
            
            rolling_corr_df = pd.DataFrame(rolling_corr)
            
            # Recent period analysis
            recent_returns = returns.tail(window)
            
            # Calculate metrics for each asset
            metrics = []
            for ticker in tickers:
                if ticker == 'SPY':
                    continue
                
                # Current correlation
                corr = calculate_correlation(
                    recent_returns[ticker].values, 
                    recent_returns['SPY'].values
                )
                
                # Up/Down Beta
                beta_up, beta_down = calculate_dual_beta(
                    recent_returns[ticker].values,
                    recent_returns['SPY'].values
                )
                
                asymmetry = beta_up - beta_down
                
                # Volatility
                vol = recent_returns[ticker].std() * np.sqrt(252) * 100
                
                # 30-day return
                cum_return = (1 + recent_returns[ticker]).prod() - 1
                
                # Correlation trend
                if len(rolling_corr_df) > 60:
                    recent_corr = rolling_corr_df[ticker].iloc[-1]
                    earlier_corr = rolling_corr_df[ticker].iloc[-60]
                    trend = "Falling" if recent_corr < earlier_corr else "Rising"
                else:
                    trend = "Stable"
                
                # Diversification score
                if abs(corr) < 0.3:
                    div_score = "High"
                elif abs(corr) < 0.7:
                    div_score = "Medium"
                else:
                    div_score = "Low"
                
                # Hedge quality
                hedge_quality = "Good" if asymmetry > 0.2 else "Poor"
                
                metrics.append({
                    'Ticker': ticker,
                    'Correlation': corr,
                    'Volatility': vol,
                    'Up Beta': beta_up,
                    'Down Beta': beta_down,
                    'Asymmetry': asymmetry,
                    'Trend': trend,
                    'Diversification': div_score,
                    'Hedge Quality': hedge_quality,
                    '30D Return': cum_return * 100,
                    'Up Days': (recent_returns['SPY'] > 0).sum(),
                    'Down Days': (recent_returns['SPY'] < 0).sum()
                })
            
            metrics_df = pd.DataFrame(metrics)
            
            # Generate insights
            insights = []
            
            # Best hedges
            good_hedges = metrics_df[metrics_df['Hedge Quality'] == 'Good']['Ticker'].tolist()
            if good_hedges:
                insights.append({
                    'type': 'high',
                    'title': '🛡️ Best Hedges Right Now',
                    'description': f"{', '.join(good_hedges)} showing asymmetric protection - they participate more in gains than losses",
                    'action': 'Consider 10-20% allocation for downside protection'
                })
            
            # Regime shifts
            for _, row in metrics_df.iterrows():
                if row['Trend'] == 'Falling' and row['Correlation'] < 0:
                    insights.append({
                        'type': 'high',
                        'title': f"📉 {row['Ticker']} Correlation Shifting Negative",
                        'description': f"{row['Ticker']} correlation with SPY is decreasing and now at {row['Correlation']:.2f} - strengthening as a hedge",
                        'action': 'Increase allocation as diversification benefit is improving'
                    })
            
            # High diversifiers
            high_div = metrics_df[metrics_df['Diversification'] == 'High']['Ticker'].tolist()
            if high_div:
                insights.append({
                    'type': 'high',
                    'title': '🎯 Strong Diversifiers',
                    'description': f"{', '.join(high_div)} have low correlation (<0.3) - excellent for reducing portfolio volatility",
                    'action': 'These assets move independently and provide true diversification'
                })
            
            # High correlation warning
            high_corr = metrics_df[metrics_df['Correlation'].abs() > 0.8]['Ticker'].tolist()
            if high_corr:
                insights.append({
                    'type': 'medium',
                    'title': '⚠️ Concentration Warning',
                    'description': f"{', '.join(high_corr)} highly correlated with SPY - limited diversification benefit",
                    'action': 'Consider reducing allocation or finding lower-correlation alternatives'
                })
            
            # Top performer
            top_performer = metrics_df.loc[metrics_df['30D Return'].idxmax()]
            insights.append({
                'type': 'low',
                'title': '📈 Top Performer',
                'description': f"{top_performer['Ticker']} returned {top_performer['30D Return']:.1f}% over the last {window} days",
                'action': 'Monitor momentum but watch for mean reversion'
            })
            
            # Display insights
            st.header("💡 Actionable Market Signals")
            for insight in insights:
                css_class = f"insight-{insight['type']}"
                st.markdown(f"""
                <div class="{css_class}">
                    <h3>{insight['title']}</h3>
                    <p>{insight['description']}</p>
                    <p style="font-style: italic; color: #94a3b8;">→ {insight['action']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Asset cards
            st.header("📊 Asset Overview")
            cols = st.columns(min(len(metrics_df), 6))
            for idx, (_, asset) in enumerate(metrics_df.iterrows()):
                with cols[idx % len(cols)]:
                    # Color coding for correlation
                    if abs(asset['Correlation']) < 0.3:
                        corr_color = "🟢"
                    elif abs(asset['Correlation']) < 0.7:
                        corr_color = "🟡"
                    else:
                        corr_color = "🔴"
                    
                    st.metric(
                        label=f"{asset['Ticker']} {corr_color}",
                        value=f"{asset['Correlation']:.2f}",
                        delta=f"{asset['30D Return']:.1f}%",
                        help=f"Correlation: {asset['Correlation']:.2f}\nVolatility: {asset['Volatility']:.1f}%"
                    )
                    st.caption(f"Vol: {asset['Volatility']:.1f}% | {asset['Diversification']} Div")
            
            # Charts
            st.header("📈 Market Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Rolling {window}-Day Correlation vs SPY")
                fig_corr = go.Figure()
                
                for ticker in rolling_corr_df.columns:
                    fig_corr.add_trace(go.Scatter(
                        x=rolling_corr_df.index,
                        y=rolling_corr_df[ticker],
                        mode='lines',
                        name=ticker,
                        line=dict(width=2.5)
                    ))
                
                fig_corr.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig_corr.update_layout(
                    template="plotly_dark",
                    height=400,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis_title="Date",
                    yaxis_title="Correlation",
                    yaxis=dict(range=[-1, 1])
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with col2:
                st.subheader("Beta Asymmetry")
                fig_beta = go.Figure()
                
                fig_beta.add_trace(go.Bar(
                    name='Up Beta',
                    x=metrics_df['Ticker'],
                    y=metrics_df['Up Beta'],
                    marker_color='rgb(16, 185, 129)'
                ))
                
                fig_beta.add_trace(go.Bar(
                    name='Down Beta',
                    x=metrics_df['Ticker'],
                    y=metrics_df['Down Beta'],
                    marker_color='rgb(239, 68, 68)'
                ))
                
                fig_beta.update_layout(
                    template="plotly_dark",
                    height=400,
                    barmode='group',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis_title="Asset",
                    yaxis_title="Beta"
                )
                st.plotly_chart(fig_beta, use_container_width=True)
            
            # Detailed metrics table
            st.header("📋 Complete Portfolio Metrics")
            
            # Format the dataframe for display
            display_df = metrics_df.copy()
            display_df['Correlation'] = display_df['Correlation'].map('{:.2f}'.format)
            display_df['Volatility'] = display_df['Volatility'].map('{:.1f}%'.format)
            display_df['Up Beta'] = display_df['Up Beta'].map('{:.2f}'.format)
            display_df['Down Beta'] = display_df['Down Beta'].map('{:.2f}'.format)
            display_df['Asymmetry'] = display_df['Asymmetry'].map('{:.2f}'.format)
            display_df['30D Return'] = display_df['30D Return'].map('{:.1f}%'.format)
            
            # Add role column
            display_df['Role'] = display_df.apply(
                lambda x: '🛡️ Hedge' if x['Hedge Quality'] == 'Good' 
                else '🎯 Diversifier' if x['Diversification'] == 'High'
                else '📈 Growth', axis=1
            )
            
            st.dataframe(
                display_df[['Ticker', 'Correlation', 'Volatility', 'Up Beta', 'Down Beta', 
                           'Asymmetry', 'Trend', '30D Return', 'Role']],
                use_container_width=True,
                height=400
            )
            
            # Key takeaways
            st.header("🎯 Portfolio Construction Framework")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success("**✅ Positive Signals**")
                st.markdown("""
                - **Falling Correlations**: Assets decoupling = Better diversification
                - **Negative Correlations**: True hedge candidates (zig when market zags)
                - **High Beta Asymmetry (>0.2)**: Convex payoff - wins bigger than losses
                - **Stable Low Correlation**: Reliable diversifier
                """)
            
            with col2:
                st.error("**⚠️ Warning Signals**")
                st.markdown("""
                - **Rising Correlations**: Crowded trades, contagion risk
                - **High Correlation (>0.8)**: Minimal diversification benefit
                - **Negative Asymmetry**: Concave payoff - losses bigger than gains
                - **Sudden Correlation Spike**: Potential regime change
                """)
            
            # Allocation framework
            st.info("""
            **💡 Suggested Allocation Framework:**
            - **Hedges (10-20%)**: Assets with negative down beta or high asymmetry
            - **Diversifiers (20-30%)**: Low correlation assets (|corr| < 0.3)
            - **Core/Satellite (50-70%)**: Market exposure balanced with defensive positions
            """)
            
            # Footer
            st.markdown("---")
            st.caption(f"""
            📊 Analysis Period: {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')} | 
            Rolling Window: {window} days | 
            Tickers: {', '.join(tickers)} | 
            Data Source: Yahoo Finance
            """)
            
        else:
            st.error("Failed to fetch data. Please check your internet connection and try again.")
