import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import coint
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns

class MarketAnalyzer:
    def __init__(self):
        # Common stock sectors and their symbols
        self.sectors = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'CRM'],
            'Finance': ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'BLK'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT'],
            'Consumer': ['AMZN', 'WMT', 'PG', 'KO', 'PEP', 'COST', 'NKE'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
        }
        self.stock_info = {}

    def fetch_market_data(self, period='1mo'):
        
        all_symbols = [symbol for symbols in self.sectors.values() for symbol in symbols]
        data = pd.DataFrame()
        
        with st.spinner('Fetching market data...'):
            for symbol in all_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    if not hist.empty:
                        data[symbol] = hist['Close']
                        
                        # Get additional info
                        info = ticker.info
                        if symbol not in self.stock_info:
                            self.stock_info[symbol] = {
                                'sector': info.get('sector', 'N/A'),
                                'market_cap': info.get('marketCap', 0),
                                'pe_ratio': info.get('trailingPE', 0),
                                'volume': info.get('averageVolume', 0),
                                'beta': info.get('beta', 0)
                            }
                except Exception as e:
                    st.warning(f"Error fetching data for {symbol}: {str(e)}")
                    continue
        
        if data.empty:
            st.error("Unable to fetch data for any symbols")
            return pd.DataFrame()
            
        return data.dropna(how='all')
    
    def calculate_market_metrics(self, data):
        """Calculate key market metrics"""
        returns = data.pct_change().dropna()
        
        metrics = pd.DataFrame(index=data.columns)
        metrics['Daily Returns Mean'] = returns.mean() * 100
        metrics['Daily Returns Std'] = returns.std() * 100
        
        # Calculate beta against the first stock in the sector
        first_stock = returns.columns[0]
        betas = []
        for col in returns.columns:
            try:
                beta = stats.linregress(returns[first_stock], returns[col])[0]
                betas.append(beta)
            except:
                betas.append(np.nan)
        metrics['Beta'] = betas
        
        metrics['Volume'] = [self.stock_info[symbol]['volume'] for symbol in data.columns]
        metrics['Market Cap (B)'] = [self.stock_info[symbol]['market_cap'] / 1e9 for symbol in data.columns]
        metrics['P/E Ratio'] = [self.stock_info[symbol]['pe_ratio'] for symbol in data.columns]
        
        return metrics
    
    def plot_correlation_matrix(self, data, sector=None):
        """Plot correlation matrix for selected sector or all stocks"""
        if sector and sector != 'All':
            symbols = self.sectors[sector]
            corr_data = data[symbols].corr()
        else:
            corr_data = data.corr()
            
        fig = px.imshow(corr_data,
                       labels=dict(x="Stock", y="Stock", color="Correlation"),
                       x=corr_data.columns,
                       y=corr_data.columns,
                       color_continuous_scale="RdBu",
                       aspect="auto")
        
        fig.update_layout(
            title=f"Correlation Matrix - {sector if sector != 'All' else 'All Stocks'}",
            width=800,
            height=800
        )
        
        return fig

class StatArbAnalyzer:
    def __init__(self, lookback_period='1y', confidence_level=0.05):
        self.lookback_period = lookback_period
        self.confidence_level = confidence_level
        
    def fetch_data(self, symbols):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        data = pd.DataFrame()
        with st.spinner('Fetching market data...'):
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=self.lookback_period)['Close']
                    data[symbol] = hist
                except Exception as e:
                    st.error(f"Error fetching data for {symbol}: {e}")
                    
        return data.dropna()
    
    def calculate_pairs_metrics(self, data):
        n = len(data.columns)
        pairs_metrics = []
        
        for i in range(n):
            for j in range(i+1, n):
                stock1 = data.columns[i]
                stock2 = data.columns[j]
                
                correlation = data[stock1].corr(data[stock2])
                score, pvalue, _ = coint(data[stock1], data[stock2])
                
                spread = data[stock1] - data[stock2]
                spread_mean = spread.mean()
                spread_std = spread.std()
                current_spread = spread.iloc[-1]
                z_score = (current_spread - spread_mean) / spread_std
                
                spread_lag = spread.shift(1)
                spread_diff = spread - spread_lag
                spread_lag = spread_lag[1:]
                spread_diff = spread_diff[1:]
                reg = np.polyfit(spread_lag, spread_diff, deg=1)
                half_life = -np.log(2) / reg[0] if reg[0] < 0 else np.inf
                
                pairs_metrics.append({
                    'stock1': stock1,
                    'stock2': stock2,
                    'correlation': correlation,
                    'cointegration_pvalue': pvalue,
                    'current_spread': current_spread,
                    'spread_mean': spread_mean,
                    'spread_std': spread_std,
                    'z_score': z_score,
                    'half_life': half_life
                })
                
        return pd.DataFrame(pairs_metrics)
    
    def plot_pair_analysis(self, data, stock1, stock2):
        # Normalize prices for comparison
        normalized_data = pd.DataFrame({
            stock1: data[stock1] / data[stock1].iloc[0] * 100,
            stock2: data[stock2] / data[stock2].iloc[0] * 100
        })
        
        # Calculate spread
        spread = data[stock1] - data[stock2]
        spread_mean = spread.mean()
        spread_std = spread.std()
        z_scores = (spread - spread_mean) / spread_std
        
        # Create subplots
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Normalized Prices', 'Z-Score of Spread'))
        
        # Plot normalized prices
        fig.add_trace(
            go.Scatter(x=normalized_data.index, y=normalized_data[stock1],
                      name=stock1, line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=normalized_data.index, y=normalized_data[stock2],
                      name=stock2, line=dict(color='red')),
            row=1, col=1
        )
        
        # Plot z-scores
        fig.add_trace(
            go.Scatter(x=z_scores.index, y=z_scores,
                      name='Z-Score', line=dict(color='green')),
            row=2, col=1
        )
        
        # Add horizontal lines for z-score thresholds
        fig.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=-2, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        fig.update_layout(height=800, title_text="Pair Analysis")
        return fig

def main():
    st.set_page_config(page_title="Market Analysis & Statistical Arbitrage", layout="wide")
    
    # Initialize analyzers
    market_analyzer = MarketAnalyzer()
    stat_arb_analyzer = StatArbAnalyzer()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Market Overview", "Statistical Arbitrage"])
    
    # Market Overview Tab
    with tab1:
        st.title("Market Overview")
        
        # Sidebar controls for Market Overview
        st.sidebar.header("Market Analysis Parameters")
        selected_period = st.sidebar.selectbox(
            "Analysis Period",
            options=['1mo', '3mo', '6mo', '1y'],
            index=0
        )
        
        selected_sector = st.sidebar.selectbox(
            "Select Sector for Correlation Analysis",
            options=['All'] + list(market_analyzer.sectors.keys())
        )
        
        # Fetch market data
        market_data = market_analyzer.fetch_market_data(period=selected_period)
        
        # Display market metrics
        st.header("Market Metrics")
        
        # Create tabs for different sectors
        sector_tabs = st.tabs(list(market_analyzer.sectors.keys()))
        
        for i, (sector, sector_tab) in enumerate(zip(market_analyzer.sectors.keys(), sector_tabs)):
            with sector_tab:
                sector_symbols = market_analyzer.sectors[sector]
                sector_data = market_data[sector_symbols]
                
                if sector_data.empty:
                    st.warning(f"No data available for {sector} sector")
                    continue
                    
                try:
                    # Display metrics table
                    sector_metrics = market_analyzer.calculate_market_metrics(sector_data)
                    
                    st.subheader(f"{sector} Sector Metrics")
                    st.dataframe(sector_metrics.style.format({
                        'Daily Returns Mean': '{:.2f}%',
                        'Daily Returns Std': '{:.2f}%',
                        'Beta': '{:.2f}',
                        'Volume': '{:,.0f}',
                        'Market Cap (B)': '{:.2f}',
                        'P/E Ratio': '{:.2f}'
                    }))
                    
                    # Display price charts
                    st.subheader("Price Performance")
                    if len(sector_data) > 0:
                        normalized_prices = sector_data.div(sector_data.iloc[0]).mul(100)
                        fig = px.line(normalized_prices, title=f"{sector} - Normalized Prices")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Insufficient data for price chart")
                        
                except Exception as e:
                    st.error(f"Error processing {sector} sector: {str(e)}")
                    continue
        
        # Display correlation matrix
        st.header("Correlation Analysis")
        corr_fig = market_analyzer.plot_correlation_matrix(market_data, selected_sector)
        st.plotly_chart(corr_fig)
        
        # Display highest correlated pairs
        st.subheader("Highest Correlated Pairs")
        corr_matrix = market_data.corr()
        pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                pairs.append({
                    'Stock 1': corr_matrix.columns[i],
                    'Stock 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        pairs_df = pd.DataFrame(pairs).sort_values('Correlation', ascending=False)
        st.dataframe(pairs_df.head(10).style.format({'Correlation': '{:.3f}'}))
    
    # Statistical Arbitrage Tab
    with tab2:
        st.title("Statistical Arbitrage Analysis")
        
        # Sidebar controls for Stat Arb
        st.sidebar.header("Statistical Arbitrage Parameters")
        
        # Input for stock symbols
        default_symbols = "AAPL,MSFT,GOOGL,META,AMZN,NFLX"
        symbols_input = st.sidebar.text_area(
            "Enter stock symbols (comma-separated)",
            value=default_symbols
        )
        symbols = [s.strip() for s in symbols_input.split(",")]
        
        # Analysis parameters
        lookback_period = st.sidebar.selectbox(
            "Lookback Period",
            options=['1mo', '3mo', '6mo', '1y', '2y'],
            index=3
        )
        
        z_score_threshold = st.sidebar.slider(
            "Z-Score Threshold",
            min_value=1.0,
            max_value=3.0,
            value=2.0,
            step=0.1
        )
        
        min_correlation = st.sidebar.slider(
            "Minimum Correlation",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05
        )
        
        # Run analysis button
        if st.sidebar.button("Run Analysis"):
            data = stat_arb_analyzer.fetch_data(symbols)
            
            if len(data.columns) < 2:
                st.error("Please enter at least 2 valid stock symbols")
                return
                
            with st.spinner('Analyzing pairs...'):
                pairs_metrics = stat_arb_analyzer.calculate_pairs_metrics(data)
                
                # Filter opportunities
                opportunities = pairs_metrics[
                    (pairs_metrics['correlation'].abs() >= min_correlation) &
                    (pairs_metrics['cointegration_pvalue'] <= stat_arb_analyzer.confidence_level) &
                    (pairs_metrics['z_score'].abs() >= z_score_threshold)
                ]
                
                # Display results
                st.header("Analysis Results")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Pairs Analyzed", len(pairs_metrics))
                with col2:
                    st.metric("Opportunities Found", len(opportunities))
                with col3:
                    st.metric("Average Correlation", f"{pairs_metrics['correlation'].mean():.3f}")
                
                # Opportunities table
                if len(opportunities) > 0:
                    st.subheader("Trading Opportunities")
                    
                    for _, opp in opportunities.iterrows():
                        with st.expander(f"{opp['stock1']} - {opp['stock2']} (Z-Score: {opp['z_score']:.2f})"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Metrics:**")
                                st.write(f"Correlation: {opp['correlation']:.3f}")
                                st.write(f"Cointegration p-value: {opp['cointegration_pvalue']:.3f}")
                                st.write(f"Half-life: {opp['half_life']:.1f} days")
                            
                            with col2:
                                st.write("**Trading Signal:**")
                                if opp['z_score'] > 0:
                                    st.write(f"Short {opp['stock1']} / Long {opp['stock2']}")
                                    st.write("Rationale: Spread is above mean by {:.2f} standard deviations".format(opp['z_score']))
                                else:
                                    st.write(f"Long {opp['stock1']} / Short {opp['stock2']}")
                                    st.write("Rationale: Spread is below mean by {:.2f} standard deviations".format(abs(opp['z_score'])))
                                
                                st.write("**Current Spread:**")
                                st.write(f"Value: {opp['current_spread']:.2f}")
                                st.write(f"Mean: {opp['spread_mean']:.2f}")
                                st.write(f"Std Dev: {opp['spread_std']:.2f}")
                            
                            # Plot pair analysis
                            fig = stat_arb_analyzer.plot_pair_analysis(data, opp['stock1'], opp['stock2'])
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # View all pairs metrics
                    with st.expander("View All Pairs Metrics"):
                        st.dataframe(pairs_metrics.sort_values('correlation', ascending=False))
                else:
                    st.info("No trading opportunities found with current parameters.")

if __name__ == '__main__':
    main()