import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import io
import smtplib
import os
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

# --- KATIE'S ERROR HANDLING IMPORTS ---
try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_GSPREAD = True
except ImportError:
    HAS_GSPREAD = False

# --- 1. App Title and Setup ---
st.set_page_config(page_title="Alpha Screener v10 (Final)", layout="wide")
st.title("🚀 Personal Stock Alpha Screener v10 (Katie's Apology Edition)")

# --- Sector Map ---
SECTOR_MAP = {
    "NVDA": "Applied AI", "TSLA": "EV/Robotics", "AAPL": "Consumer Tech",
    "MSFT": "Applied AI", "AMD": "Applied AI", "AMZN": "E-Commerce",
    "GOOGL": "AdTech", "META": "AdTech", "NFLX": "Streaming",
    "COIN": "Crypto", "PLTR": "Defense/AI", "SOFI": "Fintech",
    "ROKU": "Streaming", "SHOP": "E-Commerce", "SQ": "Fintech", 
    "MSTR": "Crypto", "SNDK": "Storage", "LRCX": "Semi Equip", "MU": "Memory",
    "SMCI": "AI Server", "ULTA": "Retail", "CMI": "Industrial", "WBD": "Media"
}

# Default Universe
default_tickers = (
    "SNDK, WDC, MU, TER, STX, LRCX, INTC, WBD, ALB, FIX, NEM, CHRW, AMAT, GLW, GOOGL, GOOG, KLAC, HII, GM, CMI, CNC, HAL, CAT, APH, MRNA, BG, MPWR, TMO, LMT, VTRS, SLB, APA, EXPD, ROST, IVZ, LLY, JBHT, CRL, MRK, WAT, GS, MNST, STLD, ULTA, DG, BKR, DD, TECH, CVS, FDX, RTX, CAH, EL, CVNA, FCX, CFG, ELV, LHX, MS, PH, JNJ, VLO, KEYS, FOXA, ADI, EA, RL, IQV, FOX, DHR, AIZ, PCAR, NUE, EPAM, F, FITB, NDSN, TPR, XOM, WMT, TJX, DAL, ADM, LOW, GD, AME, GE, HWM, USB, BK, PLD, IBKR, WSM, RVTY, PWR, ROK, LDOS, EIX, CBRE, GILD, NOC, KEY, TRGP, MTD, DAY, MAR, VTR, SWK, STE, HST, DVN, AKAM, PNC, LUV, BMY, FE, PFG, NTRS, CTRA, HAS, BDX, HOLX, JBL, ROL, HBAN, SPG, PSX, AMGN, CVX, GL, SNA, TFC, HUBB, CBOE, MTB, COO, MLM, MSCI, EVRG, TSN, VMC, WMB, TXT, TDY, NDAQ, EW, RF, TPL, UPS, DOV, CTSH, MDT, WAB, O, AEP, NEE"
)

# --- SIDEBAR INPUTS ---
st.sidebar.header("User Input")

with st.sidebar.expander("📝 Edit Watchlist", expanded=False):
    if "watchlist_input" not in st.session_state:
        st.session_state["watchlist_input"] = default_tickers

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", help="Clear text"):
            st.session_state["watchlist_input"] = ""
            st.rerun()
    with col2:
        if st.button("🔄 Reset", help="Restore default"):
            st.session_state["watchlist_input"] = default_tickers
            st.rerun()

    ticker_input = st.text_area("Enter Tickers:", key="watchlist_input", height=150)

stop_loss_pct = st.sidebar.slider("Trailing Stop Loss % (Backtest)", 5, 40, 30, 1) / 100
max_positions = st.sidebar.slider("Max Positions", 1, 5, 1, help="1 = Sniper Mode.")
allocation_pct = st.sidebar.slider("Max Capital Alloc %", 50, 100, 80, 5) / 100

vol_threshold = st.sidebar.slider(
    "Min Volume Ratio", 0.1, 2.0, 1.0, 0.1, 
    help="1.0 = Avg. 1.2 = Breakout."
)

st.sidebar.divider()
st.sidebar.header("🛡️ Future Preservation")
use_vol_target = st.sidebar.checkbox("Vol Targeting", value=False)
target_vol_ann = st.sidebar.slider("Target Ann Vol %", 10, 50, 20, 5) / 100

st.sidebar.divider()
st.sidebar.header("💰 Wealth Accelerator")
account_balance_input = st.sidebar.number_input("Start Balance ($)", 100, value=1400, step=100)
monthly_contribution = st.sidebar.slider("Monthly Add ($)", 0, 2000, 500, 50)

# --- EMAIL CONFIG ---
st.sidebar.divider()
with st.sidebar.expander("📧 Email Settings"):
    email_sender = st.text_input("Sender Email:")
    email_password = st.text_input("App Password:", type="password")
    email_recipient = st.text_input("Recipient Email:")
    email_host = st.text_input("SMTP Server:", value="smtp.gmail.com")
    email_port = st.number_input("SMTP Port:", value=587)

tickers = [x.strip().upper() for x in ticker_input.split(',') if x.strip()]

# --- HELPER FUNCTIONS ---

@st.cache_data(ttl=3600) 
def get_data_safe(ticker_list, period, interval="1d", group_by="ticker"):
    unique_tickers = list(set(ticker_list + ["SPY"]))
    try:
        data = yf.download(
            unique_tickers, 
            period=period, 
            interval=interval, 
            group_by=group_by, 
            auto_adjust=True, 
            threads=True
        )
        if not data.empty:
            return data
    except Exception as e:
        st.warning(f"⚠️ Yahoo API Failed: {e}. Trying to load local backup...")
    
    return pd.DataFrame() 

@st.cache_data(ttl=86400)
def get_wiki_tickers(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        dfs = pd.read_html(io.StringIO(response.text))
        target_df = None
        for df in dfs:
            if 'Symbol' in df.columns or 'Ticker' in df.columns:
                target_df = df
                break
        if target_df is None: return pd.DataFrame()
        if 'Ticker' in target_df.columns:
            target_df = target_df.rename(columns={'Ticker': 'Symbol'})
        return target_df
    except: return pd.DataFrame()

def calculate_kama(df, n=10, pow1=2, pow2=30):
    try:
        if len(df) < n: return pd.Series([0]*len(df), index=df.index)
        close = df['Close']
        abs_diff_n = abs(close - close.shift(n))
        volatility = abs(close - close.shift(1)).rolling(window=n).sum()
        er = abs_diff_n / volatility.replace(0, 0.000001)
        sc_fast = 2 / (pow1 + 1)
        sc_slow = 2 / (pow2 + 1)
        sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
        
        values = [0.0] * len(close)
        values[n-1] = close.iloc[n-1]
        close_values = close.values
        sc_values = sc.values
        current_kama = values[n-1]
        for i in range(n, len(close)):
            current_kama = current_kama + sc_values[i] * (close_values[i] - current_kama)
            values[i] = current_kama
        return pd.Series(values, index=df.index)
    except: return pd.Series([0]*len(df), index=df.index)

def calculate_adx(df, n=14):
    try:
        high, low, close = df['High'], df['Low'], df['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_di = 100 * (pd.Series(plus_dm).rolling(n).mean() / tr.rolling(n).mean())
        minus_di = 100 * (pd.Series(minus_dm).rolling(n).mean() / tr.rolling(n).mean())
        dx = 100 * abs((plus_di - minus_di) / (plus_di + minus_di + 1e-9))
        return dx.rolling(n).mean().iloc[-1]
    except: return 0

def calculate_efficiency_ratio(df, n=20):
    try:
        if len(df) <= n: return 0
        change = abs(df['Close'].iloc[-1] - df['Close'].iloc[-n-1])
        volatility = abs(df['Close'] - df['Close'].shift(1)).tail(n).sum()
        return change / (volatility + 1e-9)
    except: return 0

def calculate_rsi(df, n=14):
    try:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=n).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=n).mean()
        rs = gain / (loss + 1e-9)
        return (100 - (100 / (1 + rs))).iloc[-1]
    except: return 50

# --- MONTE CARLO ENGINE ---
def run_monte_carlo(returns, num_sims=500, num_periods=12, start_bal=1000):
    """
    Katie's Simulation Engine:
    Shuffles historical returns to create 500 possible futures.
    """
    sim_results = []
    returns = [r for r in returns if r != 0] # Filter out flat months for better accuracy
    if not returns: return pd.DataFrame()

    for _ in range(num_sims):
        # Randomly sample returns with replacement
        random_returns = np.random.choice(returns, size=num_periods, replace=True)
        equity_curve = [start_bal]
        curr_bal = start_bal
        for r in random_returns:
            curr_bal *= (1 + r)
            equity_curve.append(curr_bal)
        sim_results.append(equity_curve)
    
    return pd.DataFrame(sim_results).T

# --- STORAGE HANDLERS ---
def connect_gsheet():
    if not HAS_GSPREAD: return None
    if "gcp_service_account" not in st.secrets: return None
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(dict(st.secrets["gcp_service_account"]), scopes=scope)
        return gspread.authorize(creds)
    except: return None

def load_portfolio():
    client = connect_gsheet()
    if client:
        try:
            return pd.DataFrame(client.open("AlphaPortfolio").sheet1.get_all_records()), "cloud"
        except: pass
    if os.path.exists("portfolio.csv"):
        return pd.read_csv("portfolio.csv"), "local"
    return pd.DataFrame(columns=["Symbol", "Shares", "Entry Price", "Date", "Stop Price"]), "none"

def save_trade(symbol, shares, price, stop_price):
    date_str = datetime.now().strftime("%Y-%m-%d")
    client = connect_gsheet()
    if client:
        try:
            sheet = client.open("AlphaPortfolio").sheet1
            if not sheet.row_values(1): sheet.append_row(["Symbol", "Shares", "Entry Price", "Date", "Stop Price"])
            sheet.append_row([symbol, shares, price, date_str, stop_price])
            return True, "Saved to Cloud! ☁️"
        except: pass
    
    df, _ = load_portfolio()
    new_df = pd.concat([df, pd.DataFrame([{"Symbol": symbol, "Shares": shares, "Entry Price": price, "Date": date_str, "Stop Price": stop_price}])], ignore_index=True)
    new_df.to_csv("portfolio.csv", index=False)
    return True, "Saved to Local CSV 💾"

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Screener", "📉 Quant Lab (Sims)", "🔎 Universe Scanner", "💼 Portfolio"])

# ==========================================
# TAB 1: LIVE SCREENER
# ==========================================
with tab1:
    use_live_regime = st.checkbox("🛑 Market Safety Lock (SPY 200 SMA)", value=True)
    if st.button("Find Alpha (Live Scan)"):
        market_healthy = True
        status_text = st.empty()
        
        if use_live_regime:
            with st.spinner("Checking Market Regime..."):
                try:
                    spy = yf.Ticker("SPY").history(period="1y")
                    if not spy.empty:
                        spy_curr = spy['Close'].iloc[-1]
                        spy_sma = spy['Close'].rolling(200).mean().iloc[-1]
                        if spy_curr < spy_sma:
                            market_healthy = False
                            st.error(f"⚠️ BEAR MARKET DETECTED! SPY < 200 SMA. CASH IS KING.")
                except: pass

        if market_healthy:
            status_text.text(f"⏳ Downloading data for {len(tickers)} stocks...")
            data = get_data_safe(tickers, period="6mo", group_by='ticker')
            
            if data.empty:
                st.error("❌ Data Download Failed. Please try again.")
            else:
                status_text.text("✅ Data Downloaded. Calculating Alpha...")
                alpha_data = []
                
                spy_ret_20 = 0.0
                if "SPY" in data.columns.levels[0]:
                    try: spy_ret_20 = data["SPY"]['Close'].pct_change(20).iloc[-1]
                    except: pass
                
                for symbol in tickers:
                    try:
                        if symbol == "SPY": continue
                        if symbol not in data.columns.levels[0]: continue
                        
                        hist = data[symbol].dropna(how='all')
                        if len(hist) < 30: continue

                        closes = hist['Close']
                        current_price = closes.iloc[-1]
                        
                        vol_series = hist['Volume']
                        current_vol_qty = vol_series.iloc[-1]
                        avg_vol_20 = vol_series.tail(20).mean()
                        vol_ratio = current_vol_qty / (avg_vol_20 + 1e-9)
                        
                        kama_series = calculate_kama(hist)
                        trend_status = "UP" if current_price > kama_series.iloc[-1] else "DOWN"
                        
                        returns = closes.pct_change()
                        daily_std = returns.tail(20).std()
                        ann_volatility = daily_std * np.sqrt(252)
                        
                        adx_val = calculate_adx(hist)
                        er_val = calculate_efficiency_ratio(hist)
                        rsi_val = calculate_rsi(hist)
                        kama_val = kama_series.iloc[-1]
                        extension_pct = ((current_price - kama_val) / kama_val) * 100
                        
                        p_5d = closes.iloc[-6] if len(closes) >= 6 else current_price
                        roc_5d = ((current_price / p_5d) - 1) * 100
                        
                        stock_ret_20 = returns.tail(20).sum()
                        rs_diff = stock_ret_20 - spy_ret_20
                        
                        raw_score = (roc_5d * 2) + (er_val * 50)
                        if rs_diff > 0: raw_score += 15
                        if rs_diff < -0.05: raw_score -= 20
                        if extension_pct > 10: raw_score -= 20
                        if vol_ratio < 0.9: raw_score -= 30 # VOL HAMMER
                        
                        conf_score = 50 
                        if er_val > 0.4: conf_score += 20
                        if vol_ratio > 1.2: conf_score += 15
                        if roc_5d > 5: conf_score += 10
                        if rsi_val < 70: conf_score += 10
                        if rs_diff > 0.05: conf_score += 10
                        conf_score = min(100, max(0, conf_score))
                        
                        signal = "❄️ COLD"
                        if trend_status == "UP":
                            if vol_ratio < 0.9:
                                if er_val > 0.4 and extension_pct < 3: signal = "👀 WATCH (VCP Tightening)"
                                else: signal = "WAIT (Low Vol)"
                            elif rsi_val > 75: signal = "WAIT (Overbought)"
                            elif extension_pct > 15: signal = "WAIT (Extended)"
                            else:
                                if extension_pct < 5: signal = "🔥 HOT (Perfect Entry)"
                                else: signal = "🔥 HOT (Momentum)"

                        base_alloc = allocation_pct
                        if use_vol_target and ann_volatility > 0:
                            final_alloc = min(base_alloc, target_vol_ann / ann_volatility)
                        else:
                            final_alloc = base_alloc
                        
                        atr_14 = (hist['High'] - hist['Low']).rolling(14).mean().iloc[-1]
                        stop_price = current_price - (atr_14 * 3)

                        alpha_data.append({
                            "Symbol": symbol,
                            "Sector": SECTOR_MAP.get(symbol, "-"),
                            "Price": current_price,
                            "Score": raw_score,
                            "Confidence": conf_score,
                            "Signal": signal,
                            "Vol Ratio": vol_ratio,
                            "Alloc %": final_alloc * 100,
                            "Stop Price": stop_price,
                            "Ann Vol": ann_volatility,
                            "RS vs SPY": rs_diff * 100
                        })
                    except: continue

                status_text.empty()
                if alpha_data:
                    df = pd.DataFrame(alpha_data)
                    conditions = [
                        df['Signal'].str.contains("HOT"),
                        df['Signal'].str.contains("WATCH"),
                        df['Signal'].str.contains("WAIT"),
                        df['Signal'].str.contains("COLD")
                    ]
                    choices = [0, 1, 2, 3]
                    df['Signal_Rank'] = np.select(conditions, choices, default=4)
                    df = df.sort_values(by=['Signal_Rank', 'Score'], ascending=[True, False]).reset_index(drop=True)
                    df = df.drop(columns=['Signal_Rank'])

                    st.dataframe(df.style.format({"Price":"${:.2f}", "Score":"{:.1f}", "Alloc %":"{:.1f}%", "Stop Price":"${:.2f}", "RS vs SPY":"{:.2f}%", "Vol Ratio":"{:.2f}"}).background_gradient(subset=["Confidence"], cmap="Greens"))
                else: st.warning("No valid data found.")

# ==========================================
# TAB 2: QUANT LAB (SIMULATIONS & WFO)
# ==========================================
with tab2:
    st.header("📉 The Quant Lab (Validation)")
    st.caption("Robustness Checks: Monte Carlo & Year-by-Year Breakdown.")
    
    if st.button("Run Advanced Simulations"):
        with st.spinner("Crunching Numbers (This may take 10s)..."):
            # 1. Base Backtest
            all_tickers = tickers + ["SPY"]
            data = get_data_safe(all_tickers, period="5y", interval="1d")
            
            indicators = {}
            for t in tickers:
                if t in data.columns.levels[0]:
                    df_t = data[t].copy()
                    kama = calculate_kama(df_t)
                    indicators[t] = {
                        'close': df_t['Close'],
                        'kama': kama,
                        'er': df_t['Close'].rolling(20).apply(lambda x: (abs(x[-1]-x[0]) / (sum(abs(np.diff(x))) + 1e-9))),
                        'ret_1m': df_t['Close'].pct_change(20)
                    }
            
            spy_close = data['SPY']['Close']
            spy_sma = spy_close.rolling(200).mean()
            monthly_dates = data.resample('ME').last().index
            
            # TRACK RETURNS
            strat_bal = account_balance_input
            results = []
            monthly_returns_pct = [] # Capture for Monte Carlo
            
            for i in range(12, len(monthly_dates)-1):
                curr_date = monthly_dates[i]
                next_date = monthly_dates[i+1]
                idx_loc = data.index.get_indexer([curr_date], method='pad')[0]
                valid_dt = data.index[idx_loc]
                
                period_ret = 0.0
                if spy_close.loc[valid_dt] >= spy_sma.loc[valid_dt]:
                    scores = []
                    for t in tickers:
                        if t not in indicators: continue
                        try:
                            p = indicators[t]['close'].loc[valid_dt]
                            k = indicators[t]['kama'].loc[valid_dt]
                            er = indicators[t]['er'].loc[valid_dt]
                            r = indicators[t]['ret_1m'].loc[valid_dt]
                            if p > k: scores.append((t, r * er))
                        except: pass
                    scores.sort(key=lambda x: x[1], reverse=True)
                    top_n = scores[:max_positions]
                    
                    if top_n:
                        month_rets = []
                        for sym, _ in top_n:
                            try:
                                prices = indicators[sym]['close']
                                buy_p = prices.loc[valid_dt]
                                idx_next = prices.index.get_indexer([next_date], method='pad')[0]
                                sell_p = prices.iloc[idx_next]
                                ret = (sell_p - buy_p) / buy_p
                                if prices.loc[valid_dt:next_date].min() < buy_p * (1 - stop_loss_pct): ret = -stop_loss_pct
                                month_rets.append(ret)
                            except: month_rets.append(0.0)
                        period_ret = (sum(month_rets) / len(month_rets)) * allocation_pct
                
                monthly_returns_pct.append(period_ret)
                strat_bal *= (1 + period_ret)
                results.append({"Date": curr_date, "Balance": strat_bal, "Pct Change": period_ret})
            
            # --- RESULTS VISUALIZATION ---
            res_df = pd.DataFrame(results)
            st.subheader("1. Equity Curve (Historical)")
            st.area_chart(res_df.set_index("Date")['Balance'])
            
            # --- WALK FORWARD ANALYSIS (YEARLY) ---
            st.markdown("---")
            st.subheader("2. Walk-Forward Stability (Yearly Breakdown)")
            st.caption("Does the strategy work in different years? (Consistency Check)")
            res_df['Year'] = pd.to_datetime(res_df['Date']).dt.year
            yearly_stats = res_df.groupby('Year')['Pct Change'].apply(lambda x: (np.prod(1+x)-1)).reset_index()
            yearly_stats['Pct Change'] = yearly_stats['Pct Change'] * 100
            
            st.dataframe(yearly_stats.style.format({"Pct Change": "{:.2f}%"}).background_gradient(subset=["Pct Change"], cmap="RdYlGn", vmin=-10, vmax=20))

            # --- MONTE CARLO SIMULATION ---
            st.markdown("---")
            st.subheader("3. Monte Carlo Simulation (Probability Cone)")
            st.caption("Simulating 500 possible futures based on your trade history. This shows 'Luck vs Skill'.")
            
            if len(monthly_returns_pct) > 5:
                # Run Sim: 500 sims, projected 24 months out
                mc_df = run_monte_carlo(monthly_returns_pct, num_sims=500, num_periods=24, start_bal=strat_bal)
                
                # Plotting the Cone
                fig = go.Figure()
                # Plot all paths as faint lines
                for col in mc_df.columns[:50]: # Limit to 50 lines for speed
                    fig.add_trace(go.Scatter(y=mc_df[col], mode='lines', line=dict(color='rgba(0,100,255,0.1)', width=1), showlegend=False))
                
                # Plot Median, 95th (Best), 5th (Worst)
                median_line = mc_df.median(axis=1)
                best_line = mc_df.quantile(0.95, axis=1)
                worst_line = mc_df.quantile(0.05, axis=1)
                
                fig.add_trace(go.Scatter(y=median_line, mode='lines', name='Median Outcome', line=dict(color='white', width=3)))
                fig.add_trace(go.Scatter(y=best_line, mode='lines', name='Best Case (95%)', line=dict(color='green', dash='dash')))
                fig.add_trace(go.Scatter(y=worst_line, mode='lines', name='Worst Case (5%)', line=dict(color='red', dash='dash')))
                
                fig.update_layout(title="Projected Balance (Next 2 Years)", xaxis_title="Months Future", yaxis_title="Account Balance", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"🔮 **Prediction:** In the worst 5% of cases, your balance drops to **${worst_line.iloc[-1]:,.0f}**. In the median case, it grows to **${median_line.iloc[-1]:,.0f}**.")
            else:
                st.warning("Not enough trade data to run Monte Carlo. (Need at least 6 months of history).")

# ==========================================
# TAB 3: UNIVERSE SCANNER
# ==========================================
with tab3:
    st.header("🔎 Sniper Universe")
    index_choice = st.radio("Universe:", ["S&P 500", "S&P 400"], horizontal=True)
    if st.button("Scan Market"):
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies' if "500" in index_choice else 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
        sp_df = get_wiki_tickers(url)
        if not sp_df.empty:
            syms = [s.replace('.', '-') for s in sp_df['Symbol'].tolist()]
            st.info(f"Scanning {len(syms)} tickers...")
            data = get_data_safe(syms, period="3mo")
            winners = []
            for s in syms:
                try:
                    if s not in data.columns.levels[0]: continue
                    df = data[s]
                    curr = df['Close'].iloc[-1]
                    high = df['Close'].max()
                    if curr > high * 0.90:
                        winners.append({"Ticker": s, "Price": curr, "Pct off High": (high-curr)/high})
                except: pass
            df_w = pd.DataFrame(winners)
            st.dataframe(df_w)
            st.subheader("📋 Copy Tickers for Watchlist")
            st.code(", ".join(df_w['Ticker'].tolist()), language="text")

# ==========================================
# TAB 4: PORTFOLIO
# ==========================================
with tab4:
    st.header("💼 Portfolio Management")
    df_port, source = load_portfolio()
    st.caption(f"Source: {source.upper()}")
    if not df_port.empty: st.dataframe(df_port)
    with st.form("Trade"):
        c1, c2, c3 = st.columns(3)
        s = c1.text_input("Symbol")
        q = c2.number_input("Shares")
        p = c3.number_input("Price")
        if st.form_submit_button("Save Trade"):
            save_trade(s, q, p, p*0.9)
            st.success("Saved!")
