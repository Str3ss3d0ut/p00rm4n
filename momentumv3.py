import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(page_title="Sniper Alpha Hunter (GBP)", layout="wide")

# --- INITIALIZE SESSION STATE ---
if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = None
if 'scan_performed' not in st.session_state:
    st.session_state['scan_performed'] = False

# --- HELPER FUNCTIONS ---
def get_exchange_rate():
    try:
        # Get GBP to USD rate
        ticker = "GBPUSD=X"
        df = yf.download(ticker, period="1d", progress=False)
        if not df.empty:
            rate = float(df['Close'].iloc[-1].item())
            return rate
        return 1.25
    except:
        return 1.25 # Fallback average

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- BACKTESTING ENGINE ---
def run_backtest(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if len(df) < 200: return None
        
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['Vol_SMA'] = df['Volume'].rolling(window=20).mean()
        df['RVOL'] = df['Volume'] / df['Vol_SMA']
        
        df = df.dropna()
        
        in_position = False
        buy_price = 0.0
        trades = [] 
        
        close_arr = df['Close'].to_numpy().flatten()
        rsi_arr = df['RSI'].to_numpy().flatten()
        sma50_arr = df['SMA_50'].to_numpy().flatten()
        sma200_arr = df['SMA_200'].to_numpy().flatten()
        rvol_arr = df['RVOL'].to_numpy().flatten()

        for i in range(len(df)):
            price = close_arr[i]
            rsi = rsi_arr[i]
            sma50 = sma50_arr[i]
            sma200 = sma200_arr[i]
            rvol = rvol_arr[i]
            
            if not in_position:
                if (price > sma50 > sma200) and (60 < rsi < 85) and (rvol > 1.5):
                    buy_price = price
                    in_position = True
            elif in_position:
                if price < sma50:
                    profit_pct = ((price - buy_price) / buy_price) * 100
                    trades.append(profit_pct)
                    in_position = False     
        return trades
    except Exception:
        return None

# --- ROBUST TICKER FETCHING ---
@st.cache_data(ttl=86400)
def get_tickers(index_name):
    # FALLBACK LIST: If Wikipedia blocks us, use this list so the app doesn't break.
    fallback_list = [
        "NVDA", "TSLA", "AMD", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NFLX", "GME", "PLTR", 
        "COIN", "MARA", "MSTR", "HOOD", "DKNG", "UBER", "LYFT", "ABNB", "SOFI", "RIVN", "LCID",
        "F", "GM", "XOM", "CVX", "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "BA", "MMM", "CAT",
        "DE", "LMT", "RTX", "NOC", "GD", "PFE", "MRK", "JNJ", "LLY", "UNH", "CVS", "WMT", "TGT"
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    }
    
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies' if index_name == "S&P 500" else 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        html = pd.read_html(StringIO(response.text), header=0)
        tickers = html[0]['Symbol'].tolist()
        
        # Clean up tickers (replace dots with dashes for yfinance)
        clean_tickers = [t.replace('.', '-') for t in tickers]
        
        if len(clean_tickers) > 10:
            return clean_tickers
        else:
            return fallback_list # Return fallback if list is suspiciously short
            
    except Exception as e:
        # If scraping fails, fail gracefully to the fallback list
        # print(f"Scraping failed: {e}") 
        return fallback_list

# --- SCANNER LOGIC (FIXED LEAKAGE) ---
def fetch_and_scan(ticker_list, limit_num):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    tickers_to_scan = ticker_list[:limit_num]
    total = len(tickers_to_scan)
    
    for i, ticker in enumerate(tickers_to_scan):
        progress = (i + 1) / total
        progress_bar.progress(progress)
        status_text.text(f"Scanning {ticker}...")
        
        # 🛡️ CRITICAL FIX: Clear dataframe variable to prevent "Leaking" old data
        df = pd.DataFrame() 
        
        try:
            safe_ticker = ticker.replace('.', '-')
            df = yf.download(safe_ticker, period="1y", interval="1d", progress=False)
            
            if df.empty or len(df) < 200: 
                continue

            # Extract scalars safely
            current_price = float(df['Close'].iloc[-1].item())
            
            # Indicators
            sma_50 = df['Close'].rolling(window=50).mean().iloc[-1].item()
            sma_200 = df['Close'].rolling(window=200).mean().iloc[-1].item()
            df['RSI'] = calculate_rsi(df['Close'])
            rsi_current = df['RSI'].iloc[-1].item()
            
            vol_sma = df['Volume'].rolling(window=20).mean().iloc[-1].item()
            current_vol = df['Volume'].iloc[-1].item()
            rvol = current_vol / vol_sma if vol_sma > 0 else 0
            
            high_52 = df['High'].max().item()
            dist_from_high = (high_52 - current_price) / high_52
            
            high = df['High']
            low = df['Low']
            close_shift = df['Close'].shift(1)
            tr = pd.concat([high-low, (high-close_shift).abs(), (low-close_shift).abs()], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1].item()

            # Filters
            trend_perfect = current_price > sma_50 and sma_50 > sma_200
            mom_sniper = 60 < rsi_current < 85
            vol_sniper = rvol > 1.5
            near_high = dist_from_high < 0.15

            action_label = "❓"
            if rsi_current > 75: action_label = "⚠️ WAIT (Ext)"
            elif 60 <= rsi_current <= 75: action_label = "✅ BUY NOW"
            else: action_label = "✋ HOLD"

            # 🛡️ Check filters again before appending
            if trend_perfect and mom_sniper and vol_sniper and near_high:
                results.append({
                    "Ticker": safe_ticker,
                    "Action": action_label,
                    "Price": round(current_price, 2),
                    "RSI": round(rsi_current, 1),
                    "RVOL": round(rvol, 2),
                    "Dist from High": f"{round(dist_from_high * 100, 1)}%",
                    "ATR": round(atr, 2),
                    "SMA_50": round(sma_50, 2)
                })
        except Exception:
            continue
            
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(results)

# --- PORTFOLIO LOGIC ---
def track_portfolio(ticker_input):
    ticker_list = [x.strip().upper() for x in ticker_input.split(',')]
    results = []
    
    for ticker in ticker_list:
        if not ticker: continue
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            if len(df) < 50: continue
            
            current_price = float(df['Close'].iloc[-1].item())
            high = df['High']
            low = df['Low']
            close_shift = df['Close'].shift(1)
            tr = pd.concat([high-low, (high-close_shift).abs(), (low-close_shift).abs()], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1].item()
            sma_50 = df['Close'].rolling(window=50).mean().iloc[-1].item()
            stop_loss = current_price - (2 * atr)
            
            status = "✅ HEALTHY"
            if current_price < sma_50: status = "❌ BROKEN (Sell)"
            elif current_price < stop_loss: status = "⚠️ DANGER"
                
            results.append({
                "Ticker": ticker,
                "Price": round(current_price, 2),
                "Status": status,
                "Rec. Stop Loss (2x ATR)": round(stop_loss, 2),
                "Trend Floor (50 SMA)": round(sma_50, 2),
                "Volatility (ATR)": round(atr, 2)
            })
        except: continue
    return pd.DataFrame(results)

# --- APP LAYOUT ---
st.title("🇬🇧 Sniper Alpha Hunter (GBP Edition)")

# Get Live Rate
gbp_rate = get_exchange_rate()
st.sidebar.markdown(f"**Live Exchange Rate:** £1 = ${gbp_rate:.2f}")

tab1, tab2 = st.tabs(["🔭 Sniper Scanner", "💼 Portfolio Manager"])

# --- TAB 1: SCANNER ---
with tab1:
    st.markdown("### Find New Opportunities")
    with st.container():
        col_scan_1, col_scan_2 = st.columns(2)
        
        with col_scan_1:
            idx_choice = st.radio("Select Source:", ("S&P 500", "S&P 400 (MidCap)", "Both", "My Custom List"))
            
            custom_tickers = ""
            if idx_choice == "My Custom List":
                custom_tickers = st.text_area("Enter Tickers (comma separated):", "NVDA, TSLA, AMD, PLTR, GME")
            
            scan_limit = st.slider("Stocks to Scan", 10, 500, 50)

        with col_scan_2:
            st.info("Scanner looks for Price > 50 SMA, RSI 60-85, and Volume > 1.5x")
            
            if st.button("Run Sniper Scan"):
                tickers = []
                
                # LOGIC FOR SOURCE SELECTION
                if idx_choice == "My Custom List":
                     tickers = [t.strip().upper() for t in custom_tickers.split(',') if t.strip()]
                elif idx_choice == "S&P 500": 
                    tickers = get_tickers("S&P 500")
                elif idx_choice == "S&P 400 (MidCap)": 
                    tickers = get_tickers("S&P 400")
                else: 
                    # If fetching fails, these fallback lists ensure it never crashes
                    t1 = get_tickers("S&P 500")
                    t2 = get_tickers("S&P 400")
                    tickers = t1 + t2
                
                if tickers:
                    st.write(f"Sniping {len(tickers)} stocks...")
                    limit_to_use = len(tickers) if idx_choice == "My Custom List" else scan_limit
                    
                    df_results = fetch_and_scan(tickers, limit_to_use)
                    
                    if not df_results.empty:
                        df_results = df_results.sort_values(by=["Action", "RVOL"], ascending=[True, False])
                        st.session_state['scan_results'] = df_results
                        st.session_state['scan_performed'] = True
                    else:
                        st.session_state['scan_results'] = None
                        st.session_state['scan_performed'] = True
                        st.warning("No targets found in your list.")
                else:
                    st.error("Could not fetch tickers. Please check internet connection.")

    if st.session_state['scan_performed'] and st.session_state['scan_results'] is not None:
        df = st.session_state['scan_results']
        st.subheader("🎯 Sniper Targets Found")
        st.dataframe(
            df.style.map(lambda x: 'color: green; font-weight: bold;' if x == '✅ BUY NOW' else 'color: orange; font-weight: bold;' if 'WAIT' in str(x) else '', subset=['Action']),
            use_container_width=True
        )
        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🛡️ Trade Executor")
            selected_ticker = st.selectbox("Select Target:", df['Ticker'])
            row = df[df['Ticker'] == selected_ticker].iloc[0]
            price = row['Price']
            atr = row['ATR']
            sma_50 = row['SMA_50']
            action = row['Action']
            if action == "✅ BUY NOW": st.success(f"**SIGNAL:** {action}")
            else: st.warning(f"**SIGNAL:** {action}")
            st.metric("Current Price (USD)", f"${price}")
            
        with col2:
            st.subheader("💷 GBP Risk Calculator (Fractional)")
            
            account_size_gbp = st.number_input("Account Size (£)", value=10000)
            risk_pct = st.number_input("Risk %", value=1.0)
            
            account_size_usd = account_size_gbp * gbp_rate
            risk_amt_gbp = account_size_gbp * (risk_pct/100)
            risk_amt_usd = risk_amt_gbp * gbp_rate
            
            stop_price = price - (2 * atr)
            risk_per_share = price - stop_price
            
            if risk_per_share > 0:
                shares = risk_amt_usd / risk_per_share
                total_cost_usd = shares * price
                total_cost_gbp = total_cost_usd / gbp_rate
            else:
                shares = 0.0
                total_cost_usd = 0.0
                total_cost_gbp = 0.0
            
            st.info(f"""
            **Trade Plan:**
            * **Buy:** {shares:.4f} Shares
            * **Total Cost:** ${total_cost_usd:,.2f} (approx. £{total_cost_gbp:,.2f})
            * **Risk:** £{risk_amt_gbp:.2f} (If stop hit)
            * **Stop Loss:** ${stop_price:.2f}
            """)
            
            target_7pct = price * 1.07
            st.success(f"**Profit Target (Sell Half):** ${target_7pct:.2f} (+7%)")

# --- TAB 2: PORTFOLIO MANAGER ---
with tab2:
    st.markdown("### 💼 My Portfolio Tracker")
    my_tickers = st.text_input("My Tickers (comma separated):", placeholder="HAS, MAR")
    if st.button("Track My Stocks"):
        if my_tickers:
            with st.spinner("Analyzing..."):
                df_port = track_portfolio(my_tickers)
                if not df_port.empty:
                    st.dataframe(
                        df_port.style.map(lambda x: 'color: red;' if 'BROKEN' in str(x) else 'color: green;', subset=['Status']),
                        use_container_width=True
                    )
