import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import scipy.stats as si
from scipy.optimize import brentq
import re
from datetime import datetime, timezone
import glob
import os
import urllib.request
import json
import time

try:
    from curl_cffi import requests as curl_requests
    _YF_SESSION = curl_requests.Session(impersonate="chrome")
except ImportError:
    _YF_SESSION = None

def _yf_ticker(symbol):
    """Create a yf.Ticker, using a curl_cffi session when available to bypass
    Yahoo Finance bot-detection on shared IPs (e.g. Streamlit Cloud)."""
    if _YF_SESSION is not None:
        return yf.Ticker(symbol, session=_YF_SESSION)
    return yf.Ticker(symbol)

st.set_page_config(page_title="Options Tracker", layout="wide")

# PWA: manifest + service worker registration + iOS meta tags
st.markdown("""
<link rel="manifest" href="/app/static/manifest.json">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Options">
<meta name="theme-color" content="#0e1117">
<script>
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/app/static/sw.js');
  }
</script>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ── Hide Streamlit chrome ── */
#MainMenu, header, footer { display: none !important; }

/* ── Tighten page padding on mobile ── */
.block-container {
    padding-top: 0.75rem !important;
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
    max-width: 100% !important;
}

/* ── Compact tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    padding: 6px 12px !important;
    font-size: 0.82em !important;
}

/* ── Metric tiles smaller on mobile ── */
[data-testid="stMetric"] {
    padding: 6px 10px !important;
}
[data-testid="stMetricLabel"] { font-size: 0.75em !important; }
[data-testid="stMetricValue"] { font-size: 1.2em !important; }

/* ── Responsive cards: tighter padding on small screens ── */
@media (max-width: 640px) {
    .block-container {
        padding-left: 0.4rem !important;
        padding-right: 0.4rem !important;
    }
}

/* ── Expander header more compact ── */
.streamlit-expanderHeader {
    font-size: 0.88em !important;
    padding: 6px 10px !important;
}
</style>
""", unsafe_allow_html=True)

# --- BLACK-SCHOLES GREEKS ENGINE ---
def calculate_greeks(S, K, T, r, sigma, option_type):
    """Calculates Delta, Theta, and Gamma using Black-Scholes."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0, 0.0, 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Gamma is identical for calls and puts
    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))

    if option_type == 'C':
        delta = si.norm.cdf(d1)
        theta = (- (S * sigma * si.norm.pdf(d1)) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * si.norm.cdf(d2)) / 365
    else:
        delta = si.norm.cdf(d1) - 1
        theta = (- (S * sigma * si.norm.pdf(d1)) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * si.norm.cdf(-d2)) / 365

    return delta, theta, gamma

def binomial_tree_american(S, K, T, r, q, sigma, option_type, N=200):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d) 
    
    option_values = np.zeros(N + 1)
    for i in range(N + 1):
        stock_price = S * (u ** (N - i)) * (d ** i)
        if option_type == 'C':
            option_values[i] = max(0, stock_price - K)
        else:
            option_values[i] = max(0, K - stock_price)
            
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            hold_value = np.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
            current_stock_price = S * (u ** (j - i)) * (d ** i)
            if option_type == 'C':
                exercise_value = max(0, current_stock_price - K)
            else:
                exercise_value = max(0, K - current_stock_price)
            option_values[i] = max(hold_value, exercise_value)
    return option_values[0]

def black_scholes(S, K, T, r, q, sigma, option_type):
    from scipy.stats import norm
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0, S - K) if option_type == 'C' else max(0, K - S)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'C':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def implied_volatility(target_price, S, K, T, r, q, option_type):
    # Use Black-Scholes for IV solving — always monotone in sigma, no spurious roots
    def objective_function(sigma):
        return black_scholes(S, K, T, r, q, sigma, option_type) - target_price
    try:
        iv = brentq(objective_function, 0.01, 5.0)
        return iv
    except ValueError:
        return None

# Simple cache for 1m history to avoid redundant API calls per ticker
_hist_1m_cache = {}

def approximate_realtime_option_price(ticker_obj, ticker_sym, actual_stock_price, strike, last_price, trade_time, exp_date_str, opt_type):
    try:
        if ticker_sym not in _hist_1m_cache:
            hist = ticker_obj.history(period="5d", interval="1m")
            if not hist.empty:
                hist.index = hist.index.tz_convert('UTC')
            _hist_1m_cache[ticker_sym] = hist

        hist = _hist_1m_cache[ticker_sym]

        if hist.empty:
            stock_price_at_trade = actual_stock_price
        else:
            past_hist = hist[hist.index <= trade_time]
            if not past_hist.empty:
                stock_price_at_trade = past_hist.iloc[-1]['Close']
            else:
                stock_price_at_trade = actual_stock_price

        actual_stock_timestamp = datetime.now(timezone.utc)

        exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d").replace(hour=16, minute=0, second=0, tzinfo=timezone.utc)

        T_trade = max(0.001, (exp_date - trade_time).total_seconds() / (365.0 * 24 * 3600))
        T_current = max(0.001, (exp_date - actual_stock_timestamp).total_seconds() / (365.0 * 24 * 3600))

        r = 0.045
        try:
            div_yield = ticker_obj.info.get('dividendYield', 0.0) or 0.0
            q = div_yield / 100  # yfinance returns dividendYield as a percentage (e.g. 0.98 for 0.98%)
        except:
            q = 0.0

        # Solve IV from last traded price using B-S (always monotone — no spurious roots)
        true_iv = implied_volatility(last_price, stock_price_at_trade, strike, T_trade, r, q, opt_type)
        if true_iv is None:
            return 0.0, stock_price_at_trade

        return binomial_tree_american(actual_stock_price, strike, T_current, r, q, true_iv, opt_type, N=200), stock_price_at_trade
    except Exception as e:
        return 0.0, None

# --- OCC SYMBOL PARSER ---
def parse_occ(symbol):
    """Parses standard OCC option symbol into components."""
    match = re.match(r'^([A-Z]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})$', symbol.upper())
    if not match:
        return None
    ticker, yy, mm, dd, typ, strike = match.groups()
    expiration = f"20{yy}-{mm}-{dd}"
    strike_price = float(strike) / 1000.0
    return ticker, expiration, typ, strike_price

def format_occ_for_display(occ_symbol):
    """Formats a standard OCC symbol for a more readable display."""
    parsed = parse_occ(occ_symbol)
    if not parsed:
        return occ_symbol  # Return original if it's not a standard OCC, e.g., a spread name

    ticker, expiration, typ, strike_price = parsed
    exp_date = datetime.strptime(expiration, "%Y-%m-%d").strftime("%m%d%y")

    # Make strike an integer if it has no decimal part
    if strike_price == int(strike_price):
        strike_price = int(strike_price)

    return f"{ticker} {typ} {exp_date} {strike_price}"

def format_occ_html(plain_option):
    """Returns colored HTML for a plain-text formatted option string (TICKER TYPE MMDDYY STRIKE)."""
    parts = plain_option.split(' ')
    if len(parts) != 4:
        return plain_option  # spread names or unexpected format — return as-is
    ticker, typ, exp, strike = parts
    color_c = '#F472B6' if typ == 'C' else '#FB923C'  # pink for calls, orange for puts
    return (
        f'<span style="color:#60A5FA;font-weight:bold">{ticker}</span> '
        f'<span style="color:{color_c};font-weight:bold">{typ}</span> '
        f'<span style="color:#A78BFA">{exp}</span> '
        f'<span style="color:#34D399">${strike}</span>'
    )

# --- DATA FETCHING ---
def get_latest_price(ticker_obj):
    """Fetches the most up-to-date spot price and timestamp, including pre/post market."""
    hist = ticker_obj.history(period="5d", interval="1m", prepost=True)
    if not hist.empty:
        return float(hist['Close'].iloc[-1]), hist.index[-1]
    hist = ticker_obj.history(period="1d", prepost=True)
    if not hist.empty:
        return float(hist['Close'].iloc[-1]), hist.index[-1]
    return 0.0, pd.Timestamp.now(tz="UTC")

def _yf_fetch_with_retry(fn, retries=4, base_delay=8):
    """Calls fn(), retrying on rate-limit errors with exponential backoff."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if "too many requests" in str(e).lower() or "rate limit" in str(e).lower():
                if attempt < retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                else:
                    raise
            else:
                raise

@st.cache_data(ttl=300)
def fetch_option_data(occ_list):
    """Fetches delayed prices and IV from Yahoo Finance, batched by ticker."""
    results = []
    r = 0.045

    # Deduplicate OCC symbols so the same contract from multiple accounts
    # doesn't produce duplicate rows in market_data (which would cause a
    # many-to-many merge and show each position twice in the table).
    occ_list = list(dict.fromkeys(occ_list))

    # Group OCCs by ticker so we fetch each ticker's spot/chain only once
    from collections import defaultdict
    by_ticker = defaultdict(list)
    for occ in occ_list:
        parsed = parse_occ(occ)
        if parsed:
            by_ticker[parsed[0]].append((occ, parsed))

    # Increase delay between tickers on hosted environments (Streamlit Cloud
    # shares IPs, so Yahoo Finance rate-limits more aggressively there).
    inter_ticker_delay = 1.5

    for i, (ticker_sym, contracts) in enumerate(by_ticker.items()):
        if i > 0:
            time.sleep(inter_ticker_delay)
        try:
            underlying_ticker = _yf_ticker(yf_ticker(ticker_sym))
            
            def fetch_spot_and_vol(t):
                hist = t.history(period="1y", prepost=True)['Close']
                if hist.empty:
                    raise Exception("No price history")
                spot, stock_date = get_latest_price(t)
                returns = np.log(hist / hist.shift(1))
                vol = float(returns.std() * np.sqrt(252))
                return spot, vol, stock_date
                
            spot_price, hist_vol, stock_date = _yf_fetch_with_retry(
                lambda t=underlying_ticker: fetch_spot_and_vol(t)
            )
            if pd.isna(hist_vol) or hist_vol < 0.05:
                hist_vol = 0.25

            time.sleep(0.5)  # brief pause between calls within the same ticker

            # Fetch each unique expiration once per ticker
            available_exps = _yf_fetch_with_retry(
                lambda t=underlying_ticker: list(t.options)
            )
            chains = {}
            for occ, (_, expiration, _, _) in contracts:
                if expiration not in chains:
                    if expiration not in available_exps:
                        close = [e for e in available_exps if abs((datetime.strptime(e, "%Y-%m-%d") - datetime.strptime(expiration, "%Y-%m-%d")).days) <= 7]
                        st.warning(f"Expiration {expiration} not available on Yahoo Finance for {ticker_sym}. Available near this date: {close or available_exps[:5]}")
                        chains[expiration] = None
                        continue
                    try:
                        time.sleep(0.5)
                        chains[expiration] = _yf_fetch_with_retry(
                            lambda t=underlying_ticker, e=expiration: t.option_chain(e)
                        )
                    except Exception as e:
                        st.warning(f"Could not fetch chain for {ticker_sym} {expiration}: {e}")
                        chains[expiration] = None

            for occ, (_, expiration, opt_type, strike) in contracts:
                chain = chains.get(expiration)
                if chain is None:
                    continue
                try:
                    options  = chain.calls if opt_type == 'C' else chain.puts
                    contract = options[options['strike'] == strike]

                    if contract.empty:
                        available_strikes = sorted(options['strike'].tolist())
                        closest = min(available_strikes, key=lambda s: abs(s - strike)) if available_strikes else None
                        st.warning(f"Strike {strike} not in Yahoo Finance chain for {occ} (spot: {spot_price:.2f}). Closest available: {closest}. Range: {available_strikes[:3]}…{available_strikes[-3:]}")
                        continue

                    fetched_option_price = contract['lastPrice'].values[0]
                    bid = float(contract['bid'].values[0]) if 'bid' in contract.columns else 0.0
                    ask = float(contract['ask'].values[0]) if 'ask' in contract.columns else 0.0
                    last_price = (bid + ask) / 2.0 if bid > 0 and ask > 0 else fetched_option_price
                    iv         = contract['impliedVolatility'].values[0]

                    if pd.isna(iv) or iv < 0.05:
                        iv = hist_vol

                    days_to_exp = (datetime.strptime(expiration, "%Y-%m-%d") - datetime.now()).days
                    T = max(days_to_exp / 365.0, 0.001)

                    delta, theta, _ = calculate_greeks(spot_price, strike, T, r, iv, opt_type)

                    # Use approximation if option data timestamp is different from stock price timestamp
                    option_trade_time    = None
                    stock_quote_time     = stock_date
                    estimated_price      = None
                    stock_at_option_time = None
                    if 'lastTradeDate' in contract.columns:
                        trade_time = pd.to_datetime(contract['lastTradeDate'].values[0])
                        if trade_time.tzinfo is None:
                            trade_time = trade_time.tz_localize('UTC')
                        option_trade_time = trade_time
                        stock_time = stock_date
                        if stock_time.tzinfo is None:
                            stock_time = stock_time.tz_localize('UTC')

                        if trade_time < stock_time and iv > 0:
                            approx_price, stock_at_option_time = approximate_realtime_option_price(underlying_ticker, ticker_sym, spot_price, strike, last_price, trade_time, expiration, opt_type)
                            if approx_price > 0:
                                estimated_price = approx_price
                                last_price = approx_price

                    results.append({
                        "OCC_Symbol":            occ,
                        "Underlying_Price":       spot_price,
                        "Stock_Price_Timestamp":  stock_quote_time,
                        "Current_Price":          last_price,
                        "Option_Fetched_Price":   fetched_option_price,
                        "Option_Trade_Timestamp": option_trade_time,
                        "Stock_At_Option_Time":   stock_at_option_time,
                        "Estimated_Price":        estimated_price,
                        "Delta":                  delta,
                        "Theta":                  theta,
                        "DTE":                    days_to_exp
                    })
                except Exception as e:
                    st.warning(f"Could not process {occ}: {e}")

        except Exception as e:
            st.warning(f"Could not fetch data for {ticker_sym}: {e}")

    return pd.DataFrame(results)

def occ_ticker(ticker):
    """Returns the OCC-safe ticker (strips hyphens, e.g. BRK-B → BRKB)."""
    return ticker.upper().replace('-', '')

def yf_ticker(occ_tick):
    """Maps an OCC ticker back to the Yahoo Finance ticker symbol.
    Known mappings: BRKB → BRK-B. Falls back to the OCC ticker itself."""
    _MAP = {'BRKB': 'BRK-B'}
    return _MAP.get(occ_tick.upper(), occ_tick.upper())

def construct_occ_from_row(row):
    """Constructs a standard OCC option symbol from a DataFrame row."""
    ticker = occ_ticker(row['Ticker'])
    exp_str = str(row['ExpirationYYMMDD'])
    yy = exp_str[:2]
    mm = exp_str[2:4]
    dd = exp_str[4:]
    opt_type = row['OptionType']
    strike = row['Strike']
    strike_formatted = f"{int(strike * 1000):08d}"
    return f"{ticker}{yy}{mm}{dd}{opt_type.upper()}{strike_formatted}"

def save_account_to_file(all_positions_df, account):
    """Saves all positions for a given account back to its CSV file."""
    filepath = "positions/positions.csv" if account == "Default" else f"positions/positions_{account}.csv"
    cols = ['Ticker', 'ExpirationYYMMDD', 'OptionType', 'Strike', 'Side',
            'Quantity', 'Entry_Price', 'Target_Price', 'SpreadId', 'Spread_Target']
    out = all_positions_df[all_positions_df['Account'] == account][cols].copy()
    out['SpreadId'] = out['SpreadId'].fillna('')
    out['Spread_Target'] = out['Spread_Target'].fillna('')
    out.to_csv(filepath, index=False)

@st.cache_data(ttl=3600)
def parse_portfolio_pdfs():
    """Parse Robinhood portfolio statement PDFs from positions/portfolio/."""
    import pdfplumber
    import re as _re

    accounts = {}
    pdf_files = sorted(glob.glob("positions/portfolio/*.pdf"))
    if not pdf_files:
        return accounts

    acct_patterns = [
        (_re.compile(r'Individual Account #:'), 'RH', 'Individual'),
        (_re.compile(r'Traditional IRA Account #:'), 'RH-IRA', 'Traditional IRA'),
        (_re.compile(r'Roth IRA Account #:'), 'RH-Roth', 'Roth IRA'),
    ]

    for pdf_path in pdf_files:
        with pdfplumber.open(pdf_path) as pdf:
            current_key = None
            for page in pdf.pages:
                text = page.extract_text() or ""

                # Detect account boundary
                for pat, key, acct_type in acct_patterns:
                    if pat.search(text):
                        if key not in accounts:
                            accounts[key] = {
                                'name': key, 'type': acct_type,
                                'portfolio_value': None, 'cash_balance': None,
                                'total_securities': None, 'dividends_period': None,
                                'period': None, 'holdings': [],
                                'cost_basis_map': {},
                            }
                        current_key = key
                        break

                if not current_key:
                    continue
                acct = accounts[current_key]

                # Period
                if acct['period'] is None:
                    m = _re.search(r'(\d{2}/\d{2}/\d{4}) to (\d{2}/\d{2}/\d{4})', text)
                    if m:
                        acct['period'] = f"{m.group(1)} \u2013 {m.group(2)}"

                # Portfolio Value (closing)
                if acct['portfolio_value'] is None:
                    m = _re.search(r'Portfolio Value\s+\$[\d,]+\.\d+\s+\$([\d,]+\.\d+)', text)
                    if m:
                        acct['portfolio_value'] = float(m.group(1).replace(',', ''))

                # Cash balance (closing) — IRA uses "Net Account Balance", individual uses "Brokerage Cash Balance"
                if acct['cash_balance'] is None:
                    m = _re.search(r'Brokerage Cash Balance\s*\*?\s+\(?\$?([\d,]+\.\d+)\)?\s+(\(?\$?[\d,]+\.\d+\)?)', text)
                    if m:
                        raw = m.group(2).replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
                        acct['cash_balance'] = float(raw)
                    else:
                        m = _re.search(r'Net Account Balance\s+\$[\d,]+\.\d+\s+\$([\d,]+\.\d+)', text)
                        if m:
                            acct['cash_balance'] = float(m.group(1).replace(',', ''))

                # Total Securities (closing)
                if acct['total_securities'] is None:
                    m = _re.search(r'Total Securities\s*[†\*]?\s+\$[\d,]+\.\d+\s+\$([\d,]+\.\d+)', text)
                    if m:
                        acct['total_securities'] = float(m.group(1).replace(',', ''))

                # Dividends (This Period)
                if acct['dividends_period'] is None:
                    m = _re.search(r'Dividends\s+\$([\d,]+\.\d+)', text)
                    if m:
                        acct['dividends_period'] = float(m.group(1).replace(',', ''))

                # Holdings from Portfolio Summary pages — text-based (PDFs have no tables)
                if 'Portfolio Summary' in text and 'Securities Held in Account' in text:
                    lines = [l.strip() for l in text.split('\n')]
                    has_est_yield = 'Estimated Yield:' in text  # Format 2 indicator

                    _skip_pfx = ('Page ', 'Portfolio Summary', 'Securities Held in Account',
                                 'Loaned Securities', 'Total Securities', 'Brokerage Cash',
                                 'Total Priced', 'Deposit Sweep', 'Estimated Yield',
                                 '† ', '* ', '** ')

                    def _parse_mkt(raw):
                        return float(raw.replace('$', '').replace(',', '').replace('(', '-').replace(')', ''))

                    if has_est_yield:
                        # Format 2: Name line → "SYMBOL Margin QtyS? $Price ($MktValue) $EstDiv Pct%"
                        _dpat = _re.compile(
                            r'^([A-Z\.]{1,6})\s+(?:Margin|Cash)\s+([\d,]+\.?\d*)(S?)'
                            r'\s+\$([\d,]+\.\d+)\s+(\(?\$?[\d,]+\.\d+\)?)'
                            r'\s+\$[\d,]+\.\d+\s+([\d.]+)%'
                        )
                        for idx, line in enumerate(lines):
                            m = _dpat.match(line)
                            if not m:
                                continue
                            sym, qty_str, short_flag, price_str, mkt_raw, pct_str = m.groups()
                            name = ''
                            for j in range(idx - 1, max(idx - 6, -1), -1):
                                prev = lines[j]
                                if not prev or any(prev.startswith(p) for p in _skip_pfx):
                                    continue
                                if _dpat.match(prev):
                                    continue
                                name = prev
                                break
                            try:
                                is_short = bool(short_flag)
                                qty = float(qty_str.replace(',', ''))
                                if is_short:
                                    qty = -qty
                                price = float(price_str.replace(',', ''))
                                mkt_value = _parse_mkt(mkt_raw)
                                pct = float(pct_str)
                                is_option = bool(_re.search(r'\d{2}/\d{2}/\d{4}', name))
                                if sym and qty != 0:
                                    acct['holdings'].append({
                                        'name': name,
                                        'symbol': sym,
                                        'qty': qty,
                                        'price': abs(price),
                                        'mkt_value': mkt_value,
                                        'pct_portfolio': abs(pct),
                                        'is_option': is_option,
                                        'is_short': is_short,
                                        'cost_basis': None,
                                    })
                            except (ValueError, IndexError):
                                continue
                    else:
                        # Format 1: all on one line — "Name SYMBOL Margin Qty $Price $MktValue Pct%"
                        _dpat1 = _re.compile(
                            r'^(.+)\s+([A-Z\.]{1,6})\s+(?:Margin|Cash)\s+([\d,]+\.?\d*)(S?)'
                            r'\s+\$([\d,]+\.\d+)\s+(\(?\$?[\d,]+\.\d+\)?)\s+([\d.]+)%'
                        )
                        for line in lines:
                            m = _dpat1.match(line)
                            if not m:
                                continue
                            name, sym, qty_str, short_flag, price_str, mkt_raw, pct_str = m.groups()
                            try:
                                is_short = bool(short_flag)
                                qty = float(qty_str.replace(',', ''))
                                if is_short:
                                    qty = -qty
                                price = float(price_str.replace(',', ''))
                                mkt_value = _parse_mkt(mkt_raw)
                                pct = float(pct_str)
                                is_option = bool(_re.search(r'\d{2}/\d{2}/\d{4}', name))
                                if sym and qty != 0:
                                    acct['holdings'].append({
                                        'name': name.strip(),
                                        'symbol': sym,
                                        'qty': qty,
                                        'price': abs(price),
                                        'mkt_value': mkt_value,
                                        'pct_portfolio': abs(pct),
                                        'is_option': is_option,
                                        'is_short': is_short,
                                        'cost_basis': None,
                                    })
                            except (ValueError, IndexError):
                                continue

                # Cost basis from Gain/Loss section
                if current_key and ('Cost Basis' in text or 'Unrealized' in text):
                    acct = accounts[current_key]
                    for table in (page.extract_tables() or []):
                        if not table:
                            continue
                        cost_col = sym_col = header_idx = None
                        for i, row in enumerate(table):
                            if not row:
                                continue
                            row_strs = [str(c or '').strip() for c in row]
                            if any('Cost Basis' in c for c in row_strs):
                                header_idx = i
                                for j, cell in enumerate(row_strs):
                                    if 'Cost Basis' in cell:
                                        cost_col = j
                                    if cell in ('Symbol', 'Sym', 'Sym/Cusip', 'Sym/ Cusip'):
                                        sym_col = j
                                break
                        if cost_col is None:
                            continue
                        if sym_col is None:
                            sym_col = 1
                        data_rows = table[header_idx + 1:] if header_idx is not None else table
                        for row in data_rows:
                            if not row:
                                continue
                            try:
                                sym = str(row[sym_col] or '').strip() if sym_col < len(row) else ''
                                cost_raw = str(row[cost_col] or '').strip() if cost_col < len(row) else ''
                                if not sym or not cost_raw or sym in ('', 'None', 'Symbol', 'Sym/Cusip'):
                                    continue
                                cost_clean = cost_raw.replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
                                acct['cost_basis_map'][sym] = float(cost_clean)
                            except (ValueError, IndexError):
                                continue

    # Match cost_basis_map entries to holdings
    for acct in accounts.values():
        cb_map = acct.get('cost_basis_map', {})
        if cb_map:
            for holding in acct['holdings']:
                if holding['cost_basis'] is None and holding['symbol'] in cb_map:
                    holding['cost_basis'] = cb_map[holding['symbol']]

    return accounts

# --- MAIN DASHBOARD ---
st.markdown(
    '<div style="font-size:1.1em;font-weight:700;color:#e2e8f0;'
    'padding:4px 0 10px;letter-spacing:0.01em;">Options Tracker</div>',
    unsafe_allow_html=True
)
page_tab1, page_tab2, page_tab3, page_tab4, page_tab5 = st.tabs(["Portfolio", "Watchlist", "Sentiment", "Summary", "Trades"])

with page_tab1:
    # Load Positions from multiple files
    try:
        position_files = glob.glob("positions/positions_*.csv")
        all_positions = []

        if os.path.exists("positions/positions.csv"):
            position_files.append("positions/positions.csv")

        if not position_files:
            st.error("No position files found (e.g., positions.csv, positions_FD.csv).")
            st.stop()

        for file in position_files:
            account_name = os.path.basename(file).replace("positions_", "").replace(".csv", "")
            if account_name == "positions":
                account_name = "Default"

            column_names = [
                'Ticker', 'ExpirationYYMMDD', 'OptionType', 'Strike', 'Side',
                'Quantity', 'Entry_Price', 'Target_Price', 'SpreadId', 'Spread_Target'
            ]
            df = pd.read_csv(file, header=None, skiprows=1, names=column_names, comment='#')
            df['Account'] = account_name
            all_positions.append(df)

        positions = pd.concat(all_positions, ignore_index=True)

        if 'SpreadId' not in positions.columns:
            positions['SpreadId'] = np.nan
        if 'Spread_Target' not in positions.columns:
            positions['Spread_Target'] = np.nan

        positions['OCC_Symbol'] = positions.apply(construct_occ_from_row, axis=1)

    except Exception as e:
        st.error(f"An error occurred while reading position files: {e}")
        st.stop()

    # Live mode toggle for Portfolio
    if 'pf_live' not in st.session_state:
        st.session_state['pf_live'] = False
    if 'pf_interval' not in st.session_state:
        st.session_state['pf_interval'] = 30
    if 'pf_sort' not in st.session_state:
        st.session_state['pf_sort'] = 'Expiry'

    pf_ctrl1, pf_ctrl2, pf_ctrl3, pf_ctrl4 = st.columns([2, 2, 2, 2])
    pf_live_on = pf_ctrl1.toggle("Live Quotes", value=st.session_state['pf_live'], key="pf_live_toggle")
    st.session_state['pf_live'] = pf_live_on
    if pf_live_on:
        pf_interval = pf_ctrl2.selectbox("Refresh every", [15, 30, 60, 120], index=1, format_func=lambda x: f"{x}s", key="pf_interval_sel")
        st.session_state['pf_interval'] = pf_interval
    pf_sort = pf_ctrl3.selectbox("Sort by", ["Expiry", "Symbol"], index=0 if st.session_state['pf_sort'] == 'Expiry' else 1, key="pf_sort_sel")
    st.session_state['pf_sort'] = pf_sort

    # Account filter
    accounts = ["All"] + positions['Account'].unique().tolist()
    selected_accounts = st.multiselect("Filter by Account", options=accounts, default=["All"])

    if "All" in selected_accounts:
        filtered_positions = positions
    else:
        filtered_positions = positions[positions['Account'].isin(selected_accounts)]

    if pf_live_on:
        fetch_option_data.clear()

    with st.spinner("Fetching latest market data..."):
        market_data = fetch_option_data(filtered_positions['OCC_Symbol'].tolist())

    if not market_data.empty:
        df = pd.merge(filtered_positions, market_data, on="OCC_Symbol", how="left")

        df['SpreadId'] = df['SpreadId'].fillna('')
        singles_df = df[df['SpreadId'] == ''].copy()
        spreads_df = df[df['SpreadId'] != ''].copy()

        processed_positions = []

        # --- PROCESS SINGLES ---
        if not singles_df.empty:
            singles_df['Unrealized_P&L_$'] = np.where(
                singles_df['Side'].str.upper() == 'LONG',
                (singles_df['Current_Price'] - singles_df['Entry_Price']) * 100 * singles_df['Quantity'],
                (singles_df['Entry_Price'] - singles_df['Current_Price']) * 100 * singles_df['Quantity']
            )
            singles_df['P&L_%'] = (singles_df['Unrealized_P&L_$'] / (singles_df['Entry_Price'] * 100 * singles_df['Quantity'])).fillna(0) * 100

            singles_df['Position_Delta'] = np.where(
                singles_df['Side'].str.upper() == 'SHORT',
                -singles_df['Delta'],
                singles_df['Delta']
            )
            singles_df['Position_Theta'] = np.where(
                singles_df['Side'].str.upper() == 'SHORT',
                -singles_df['Theta'],
                singles_df['Theta']
            )
            singles_df['Price_Diff_To_Target'] = np.where(
                singles_df['Side'].str.upper() == 'LONG',
                singles_df['Target_Price'] - singles_df['Current_Price'],
                singles_df['Current_Price'] - singles_df['Target_Price']
            )
            safe_theta = np.where(np.abs(singles_df['Position_Theta']) > 1e-4, singles_df['Position_Theta'], np.nan)
            singles_df['Days_To_Target_(Theta)'] = singles_df['Price_Diff_To_Target'] / safe_theta

            safe_delta = np.where(np.abs(singles_df['Position_Delta']) > 1e-4, singles_df['Position_Delta'], np.nan)
            
            intrinsic_target_stock = np.where(
                singles_df['OptionType'].str.upper() == 'C',
                singles_df['Strike'] + singles_df['Target_Price'],
                singles_df['Strike'] - singles_df['Target_Price']
            )

            linear_move = singles_df['Price_Diff_To_Target'] / safe_delta
            linear_target = singles_df['Underlying_Price'] + linear_move

            is_long_call = (singles_df['Side'].str.upper() == 'LONG') & (singles_df['OptionType'].str.upper() == 'C')
            is_long_put = (singles_df['Side'].str.upper() == 'LONG') & (singles_df['OptionType'].str.upper() == 'P')

            capped_target = np.where(
                is_long_call,
                np.minimum(linear_target, intrinsic_target_stock),
                np.where(
                    is_long_put,
                    np.maximum(linear_target, intrinsic_target_stock),
                    linear_target
                )
            )

            final_target = np.where(np.isnan(safe_delta), intrinsic_target_stock, capped_target)

            singles_df['Underlying_Move_Needed_$'] = final_target - singles_df['Underlying_Price']
            singles_df['Target_Hit'] = np.where(
                singles_df['Side'].str.upper() == 'LONG',
                singles_df['Current_Price'] >= singles_df['Target_Price'],
                singles_df['Current_Price'] <= singles_df['Target_Price']
            )
            singles_df['Underlying_Target'] = final_target
            processed_positions.append(singles_df)

        # --- PROCESS SPREADS ---
        if not spreads_df.empty:
            aggregated_spreads = []
            for spread_id, group in spreads_df.groupby('SpreadId'):
                if group.empty: continue

                net_entry_price = 0
                net_current_price = 0
                for _, leg in group.iterrows():
                    if leg['Side'].upper() == 'SHORT':
                        net_entry_price += leg['Entry_Price']
                        net_current_price += leg['Current_Price']
                    else:
                        net_entry_price -= leg['Entry_Price']
                        net_current_price -= leg['Current_Price']

                is_credit_spread = net_entry_price > 0
                side = "CREDIT" if is_credit_spread else "DEBIT"
                spread_target = group['Spread_Target'].dropna().iloc[0] if not group['Spread_Target'].dropna().empty else 0

                ticker = group['Ticker'].iloc[0]
                exp = group['ExpirationYYMMDD'].iloc[0]
                strikes = "/".join(map(str, sorted(group['Strike'].astype(int).tolist())))
                opt_type = group['OptionType'].iloc[0]
                spread_name = f"{ticker} {exp} {strikes}{opt_type}"

                pnl = (net_entry_price - net_current_price) * 100 * group['Quantity'].iloc[0]

                current_spread_value = net_current_price if is_credit_spread else -net_current_price

                spread_delta = 0.0
                spread_theta = 0.0
                for _, leg in group.iterrows():
                    sign = 1 if leg['Side'].upper() == 'SHORT' else -1
                    if not is_credit_spread:
                        sign = -sign
                    spread_delta += sign * leg['Delta']
                    spread_theta += sign * leg['Theta']

                spread_price_change_needed = spread_target - current_spread_value
                days_to_target = spread_price_change_needed / spread_theta if abs(spread_theta) > 1e-4 else np.nan
                underlying_move = spread_price_change_needed / spread_delta if abs(spread_delta) > 1e-4 else np.nan
                
                if is_credit_spread:
                    target_hit = current_spread_value <= spread_target
                else:
                    target_hit = current_spread_value >= spread_target

                aggregated_spreads.append({
                    'Account': group['Account'].iloc[0],
                    'OCC_Symbol': spread_name,
                    'Underlying_Price': group['Underlying_Price'].iloc[0],
                    'Underlying_Target': group['Underlying_Price'].iloc[0] + underlying_move if pd.notna(underlying_move) else np.nan,
                    'Side': side,
                    'Quantity': group['Quantity'].iloc[0],
                    'Entry_Price': net_entry_price,
                    'Current_Price': net_current_price,
                    'Target_Price': spread_target,
                    'Unrealized_P&L_$': pnl,
                    'P&L_%': np.nan,
                    'Days_To_Target_(Theta)': days_to_target,
                    'Underlying_Move_Needed_$': underlying_move,
                    'Target_Hit': target_hit
                })

            if aggregated_spreads:
                processed_positions.append(pd.DataFrame(aggregated_spreads))

        # --- COMBINE AND DISPLAY ---
        if not processed_positions:
            st.warning("No positions to display.")
            st.stop()

        display_df = pd.concat(processed_positions, ignore_index=True)

        # Extract sort keys before OCC_Symbol is converted to HTML
        def _occ_sort_keys(sym):
            parsed = parse_occ(sym)
            if parsed:
                return parsed[0], parsed[1]  # ticker, YYYY-MM-DD expiry
            # Spread names like "META 260529 720/700P" — parts[0]=ticker, parts[1]=YYMMDD
            parts = sym.split()
            ticker_part = parts[0] if parts else sym
            exp_part = f"20{parts[1][:2]}-{parts[1][2:4]}-{parts[1][4:]}" if len(parts) > 1 and len(parts[1]) == 6 else parts[1] if len(parts) > 1 else ''
            return ticker_part, exp_part

        sort_keys = display_df['OCC_Symbol'].apply(_occ_sort_keys)
        display_df['_sort_ticker'] = sort_keys.apply(lambda x: x[0])
        display_df['_sort_expiry'] = sort_keys.apply(lambda x: x[1])

        if st.session_state.get('pf_sort', 'Expiry') == 'Symbol':
            display_df = display_df.sort_values(['Account', '_sort_ticker', '_sort_expiry']).reset_index(drop=True)
        else:
            display_df = display_df.sort_values(['Account', '_sort_expiry', '_sort_ticker']).reset_index(drop=True)

        display_df['OCC_Symbol'] = display_df['OCC_Symbol'].apply(format_occ_for_display).apply(format_occ_html)

        display_df = display_df[[
            'Account', 'OCC_Symbol', 'Underlying_Price', 'Underlying_Target', 'Side', 'Quantity', 'Entry_Price', 'Current_Price', 'Target_Price',
            'Unrealized_P&L_$', 'P&L_%', 'Days_To_Target_(Theta)', 'Underlying_Move_Needed_$', 'Target_Hit'
        ]].copy()
        display_df = display_df.rename(columns={'OCC_Symbol': 'Option'})

        display_df = display_df.round({
            'Current_Price': 2, 'Unrealized_P&L_$': 2, 'P&L_%': 2,
            'Days_To_Target_(Theta)': 1, 'Underlying_Move_Needed_$': 2
        })

        total_pnl = display_df['Unrealized_P&L_$'].sum()
        st.metric(label="Total Portfolio Unrealized P&L", value=f"${total_pnl:,.2f}")

        def position_card_html(row):
            option_html  = row['Option']   # already colored HTML
            side         = str(row['Side'])
            qty          = int(row['Quantity']) if pd.notna(row['Quantity']) else 1
            entry        = row['Entry_Price']
            current      = row['Current_Price']
            target       = row['Target_Price']
            underlying   = row['Underlying_Price']
            u_target     = row['Underlying_Target']
            pnl          = row['Unrealized_P&L_$']
            pnl_pct      = row['P&L_%']
            dte          = row['Days_To_Target_(Theta)']
            move         = row['Underlying_Move_Needed_$']
            target_hit   = bool(row['Target_Hit'])

            # Colors
            border_color = '#22c55e' if target_hit else ('#f59e0b' if side.upper() in ('CREDIT', 'SHORT') else '#6366f1')
            bg_color     = 'rgba(34,197,94,0.07)' if target_hit else 'rgba(255,255,255,0.03)'
            pnl_color    = '#22c55e' if pd.notna(pnl) and pnl >= 0 else '#f87171'
            side_color   = '#f87171' if side.upper() in ('SHORT', 'CREDIT') else '#34d399'
            price_color  = '#22c55e' if target_hit else '#e2e8f0'

            badge = '<span style="background:#22c55e;color:#000;font-size:0.6em;padding:2px 7px;border-radius:4px;font-weight:bold;vertical-align:middle;">TARGET HIT</span>' if target_hit else ''

            entry_str   = f"${entry:.2f}"   if pd.notna(entry)   else "—"
            current_str = f"${current:.2f}" if pd.notna(current) else "—"
            target_str  = f"${target:.2f}"  if pd.notna(target)  else "—"
            pnl_str     = f"${pnl:+,.2f}"   if pd.notna(pnl)     else "—"
            pnl_pct_str = f"{pnl_pct:+.1f}%" if pd.notna(pnl_pct) and pnl_pct == pnl_pct else ""
            dte_str     = f"{dte:.0f}d"      if pd.notna(dte) and dte == dte else "—"
            move_str    = f"${move:+.2f}"    if pd.notna(move) and move == move else "—"
            spot_str    = f"${underlying:.2f}" if pd.notna(underlying) else "—"
            u_target_str = f"${u_target:.2f}" if pd.notna(u_target) else "—"

            badge_html   = ('&nbsp;' + badge) if target_hit else ''

            return (
                f'<div style="border:1.5px solid {border_color};border-radius:12px;padding:10px 14px;'
                f'background:{bg_color};display:flex;justify-content:space-between;gap:10px;">'

                # LEFT
                f'<div style="flex:1;min-width:0;">'
                f'<div style="font-size:0.9em;font-weight:600;line-height:1.5;">{option_html}</div>'
                f'<div style="display:flex;gap:10px;margin-top:6px;flex-wrap:wrap;">'
                f'<div><div style="color:#6B7280;font-size:0.6em;text-transform:uppercase;">Entry</div>'
                f'<div style="color:#9CA3AF;font-size:0.82em;font-weight:600;">{entry_str}</div></div>'
                f'<div><div style="color:#6B7280;font-size:0.6em;text-transform:uppercase;">Days</div>'
                f'<div style="color:#e2e8f0;font-size:0.82em;font-weight:600;">{dte_str}</div></div>'
                f'<div><div style="color:#6B7280;font-size:0.6em;text-transform:uppercase;">Move</div>'
                f'<div style="color:#60A5FA;font-size:0.82em;font-weight:600;">{move_str}</div></div>'
                f'<div><div style="color:#6B7280;font-size:0.6em;text-transform:uppercase;">Spot Tgt</div>'
                f'<div style="color:#A78BFA;font-size:0.82em;font-weight:600;">{spot_str} &rarr; {u_target_str}</div></div>'
                f'</div></div>'

                # RIGHT
                f'<div style="text-align:right;flex-shrink:0;display:flex;flex-direction:column;justify-content:space-between;">'
                f'<div>'
                f'<div style="color:#6B7280;font-size:0.6em;text-transform:uppercase;">'
                f'<span style="color:{side_color};font-weight:700;">{side}</span> x{qty}{badge_html}</div>'
                f'<div style="color:{price_color};font-size:1.15em;font-weight:700;line-height:1.2;">{current_str}</div>'
                f'<div style="color:#C084FC;font-size:0.85em;font-weight:700;">&#8594; {target_str}</div>'
                f'</div>'
                f'<div style="margin-top:6px;">'
                f'<div style="color:{pnl_color};font-size:0.95em;font-weight:700;">{pnl_str}</div>'
                f'<div style="color:{pnl_color};font-size:0.7em;opacity:0.8;">{pnl_pct_str}</div>'
                f'</div>'
                f'</div>'
                f'</div>'
            )

        # Group cards by account
        for account in display_df['Account'].unique():
            acct_df  = display_df[display_df['Account'] == account]
            acct_pnl = acct_df['Unrealized_P&L_$'].sum()
            pnl_col  = '#22c55e' if acct_pnl >= 0 else '#f87171'
            st.markdown(
                f'<div style="margin:18px 0 6px;font-size:0.9em;font-weight:700;color:#9CA3AF;'
                f'text-transform:uppercase;letter-spacing:0.05em;">'
                f'{account} &nbsp;<span style="color:{pnl_col};font-size:0.95em;">${acct_pnl:+,.2f}</span></div>',
                unsafe_allow_html=True
            )
            cards_html = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:8px;width:100%;">'
            for _, row in acct_df.iterrows():
                cards_html += position_card_html(row)
            cards_html += '</div>'
            st.markdown(cards_html, unsafe_allow_html=True)

        pf_updated_str = datetime.now().strftime("%H:%M:%S")
        pf_status_color = '#22c55e' if pf_live_on else '#6B7280'
        pf_live_badge = (
            f'<span style="background:{pf_status_color};color:#000;font-size:0.65em;'
            f'padding:2px 8px;border-radius:10px;font-weight:700;">{"LIVE" if pf_live_on else "DELAYED"}</span>'
            f'&nbsp;<span style="color:#6B7280;font-size:0.72em;">Updated {pf_updated_str}</span>'
        )
        st.markdown(pf_live_badge, unsafe_allow_html=True)
        st.caption("Greeks are Black-Scholes approximations.")

        with st.expander("Price Details", expanded=False):
            if not market_data.empty:
                detail_cols = ['OCC_Symbol', 'Option_Fetched_Price', 'Option_Trade_Timestamp',
                               'Stock_At_Option_Time', 'Underlying_Price', 'Stock_Price_Timestamp',
                               'Estimated_Price', 'Current_Price']
                available = [c for c in detail_cols if c in market_data.columns]
                detail_df = market_data[available].copy()

                def _fmt_ts(ts):
                    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
                        return "—"
                    try:
                        ts = pd.Timestamp(ts)
                        if ts.tzinfo is not None:
                            ts = ts.tz_convert('US/Eastern')
                        return ts.strftime("%Y-%m-%d %H:%M ET")
                    except Exception:
                        return str(ts)

                if 'Option_Trade_Timestamp' in detail_df.columns:
                    detail_df['Option_Trade_Timestamp'] = detail_df['Option_Trade_Timestamp'].apply(_fmt_ts)
                if 'Stock_Price_Timestamp' in detail_df.columns:
                    detail_df['Stock_Price_Timestamp'] = detail_df['Stock_Price_Timestamp'].apply(_fmt_ts)
                if 'Estimated_Price' in detail_df.columns:
                    detail_df['Estimated_Price'] = detail_df['Estimated_Price'].apply(
                        lambda x: f"${x:.2f}" if pd.notna(x) and x is not None else "—"
                    )

                detail_df['OCC_Symbol'] = detail_df['OCC_Symbol'].apply(format_occ_for_display)
                detail_df = detail_df.rename(columns={
                    'OCC_Symbol':             'Option',
                    'Option_Fetched_Price':   'Fetched Price',
                    'Option_Trade_Timestamp': 'Option Timestamp',
                    'Stock_At_Option_Time':   'Stock @ Option Time',
                    'Underlying_Price':       'Stock Price (now)',
                    'Stock_Price_Timestamp':  'Stock Timestamp',
                    'Estimated_Price':        'Est. Price',
                    'Current_Price':          'Used Price',
                })
                for col in ['Fetched Price', 'Stock @ Option Time', 'Stock Price (now)', 'Used Price']:
                    if col in detail_df.columns:
                        detail_df[col] = detail_df[col].apply(
                            lambda x: f"${x:.2f}" if pd.notna(x) else "—"
                        )
                st.dataframe(detail_df, use_container_width=True, hide_index=True)

        if st.button("Refresh Market Data"):
            fetch_option_data.clear()
            st.rerun()

        if pf_live_on:
            pf_countdown = st.empty()
            for i in range(st.session_state['pf_interval'], 0, -1):
                pf_countdown.markdown(
                    f'<span style="color:#6B7280;font-size:0.75em;">Next refresh in {i}s</span>',
                    unsafe_allow_html=True
                )
                time.sleep(1)
            pf_countdown.empty()
            fetch_option_data.clear()
            st.rerun()

    # --- MANAGE POSITIONS ---
    st.divider()
    with st.expander("Manage Positions", expanded=False):
        mp_add, mp_edit, mp_delete = st.tabs(["Add", "Edit", "Delete"])

        account_options = sorted(set(
            [os.path.basename(f).replace("positions_", "").replace(".csv", "")
             for f in glob.glob("positions/positions_*.csv")]
            + (["Default"] if os.path.exists("positions/positions.csv") else [])
        ))

        def pos_label(row):
            spread = f" [{row['SpreadId']}]" if pd.notna(row['SpreadId']) and str(row['SpreadId']).strip() else ""
            return f"{row['Ticker']} {row['ExpirationYYMMDD']} {row['OptionType']} {row['Strike']} {row['Side']}{spread} ({row['Account']})"

        with mp_add:
            with st.form("add_position_form"):
                c1, c2, c3, c4 = st.columns(4)
                ticker_in   = c1.text_input("Ticker", placeholder="NVDA").strip().upper()
                exp_in      = c2.text_input("Expiry YYMMDD", placeholder="270115")
                opt_type_in = c3.selectbox("Type", ["P", "C"])
                strike_in   = c4.number_input("Strike", min_value=0.01, step=0.5, value=100.0)

                c5, c6, c7, c8 = st.columns(4)
                side_in    = c5.selectbox("Side", ["Short", "Long"])
                qty_in     = c6.number_input("Quantity", min_value=1, step=1, value=1)
                entry_in   = c7.number_input("Entry Price", min_value=0.01, step=0.01, value=1.0)
                target_in  = c8.number_input("Target Price", min_value=0.0, step=0.01, value=0.0)

                c9, c10, c11 = st.columns(3)
                account_in       = c9.selectbox("Account", account_options) if account_options else c9.text_input("Account")
                spread_id_in     = c10.text_input("Spread ID (optional)")
                spread_target_in = c11.number_input("Spread Target (optional)", min_value=0.0, step=0.01, value=0.0)

                if st.form_submit_button("Add Position"):
                    if not ticker_in or not exp_in:
                        st.error("Ticker and Expiry are required.")
                    else:
                        new_row = {
                            'Ticker': ticker_in, 'ExpirationYYMMDD': int(exp_in),
                            'OptionType': opt_type_in, 'Strike': float(strike_in),
                            'Side': side_in, 'Quantity': int(qty_in),
                            'Entry_Price': float(entry_in),
                            'Target_Price': float(target_in) if target_in else np.nan,
                            'SpreadId': spread_id_in.strip() if spread_id_in.strip() else np.nan,
                            'Spread_Target': float(spread_target_in) if spread_target_in else np.nan,
                            'Account': account_in,
                        }
                        updated = pd.concat([positions, pd.DataFrame([new_row])], ignore_index=True)
                        save_account_to_file(updated, account_in)
                        fetch_option_data.clear()
                        st.success(f"Added {ticker_in} {opt_type_in} {exp_in} ${strike_in} to {account_in}.")
                        st.rerun()

        with mp_edit:
            labels = [pos_label(r) for _, r in positions.iterrows()]
            if not labels:
                st.info("No positions to edit.")
            else:
                sel_label = st.selectbox("Select position", labels, key="edit_sel")
                sel_idx   = labels.index(sel_label)
                row       = positions.iloc[sel_idx]

                with st.form("edit_position_form"):
                    c1, c2, c3, c4 = st.columns(4)
                    ticker_e    = c1.text_input("Ticker", value=str(row['Ticker'])).strip().upper()
                    exp_e       = c2.text_input("Expiry YYMMDD", value=str(row['ExpirationYYMMDD']))
                    opt_type_e  = c3.selectbox("Type", ["P", "C"], index=0 if row['OptionType'] == 'P' else 1)
                    strike_e    = c4.number_input("Strike", value=float(row['Strike']), step=0.5)

                    c5, c6, c7, c8 = st.columns(4)
                    side_e   = c5.selectbox("Side", ["Short", "Long"], index=0 if str(row['Side']).lower() == 'short' else 1)
                    qty_e    = c6.number_input("Quantity", value=int(row['Quantity']), min_value=1, step=1)
                    entry_e  = c7.number_input("Entry Price", value=float(row['Entry_Price']), step=0.01)
                    target_e = c8.number_input("Target Price",
                                   value=float(row['Target_Price']) if pd.notna(row['Target_Price']) else 0.0,
                                   step=0.01)

                    c9, c10, c11 = st.columns(3)
                    spread_id_e     = c9.text_input("Spread ID", value=str(row['SpreadId']) if pd.notna(row['SpreadId']) else '')
                    spread_target_e = c10.number_input("Spread Target",
                                          value=float(row['Spread_Target']) if pd.notna(row['Spread_Target']) else 0.0,
                                          step=0.01)

                    if st.form_submit_button("Save Changes"):
                        positions.at[sel_idx, 'Ticker']           = ticker_e
                        positions.at[sel_idx, 'ExpirationYYMMDD'] = int(exp_e)
                        positions.at[sel_idx, 'OptionType']       = opt_type_e
                        positions.at[sel_idx, 'Strike']           = float(strike_e)
                        positions.at[sel_idx, 'Side']             = side_e
                        positions.at[sel_idx, 'Quantity']         = int(qty_e)
                        positions.at[sel_idx, 'Entry_Price']      = float(entry_e)
                        positions.at[sel_idx, 'Target_Price']     = float(target_e) if target_e else np.nan
                        positions.at[sel_idx, 'SpreadId']         = spread_id_e.strip() if spread_id_e.strip() else np.nan
                        positions.at[sel_idx, 'Spread_Target']    = float(spread_target_e) if spread_target_e else np.nan
                        save_account_to_file(positions, row['Account'])
                        fetch_option_data.clear()
                        st.success("Position updated.")
                        st.rerun()

        with mp_delete:
            labels_del = [pos_label(r) for _, r in positions.iterrows()]
            if not labels_del:
                st.info("No positions to delete.")
            else:
                sel_del     = st.selectbox("Select position to delete", labels_del, key="del_sel")
                sel_del_idx = labels_del.index(sel_del)
                acct_del    = positions.iloc[sel_del_idx]['Account']

                st.warning(f"This will permanently remove: **{sel_del}**")
                if st.button("Delete Position", type="primary"):
                    updated = positions.drop(index=sel_del_idx).reset_index(drop=True)
                    save_account_to_file(updated, acct_del)
                    fetch_option_data.clear()
                    st.success("Position deleted.")
                    st.rerun()

# =====================================================================
# --- WATCHLIST TAB ---
# =====================================================================
WATCHLIST_FILE = "positions/watchlist.csv"
WATCHLIST_COLS = ['Ticker', 'ExpirationYYMMDD', 'OptionType', 'Strike', 'TargetPrice', 'Intent', 'Label', 'ItemType']

TRADES_FILE = "positions/trades.csv"
TRADES_COLS = ['id', 'date', 'ticker', 'strategy', 'event_type', 'qty',
               'strike', 'expiry', 'option_type', 'price', 'fees',
               'account_type', 'capital_reserved', 'notes', 'leg']

TRADE_EVENTS = [
    "Buy Stock", "Sell Stock",
    "Sell Covered Call", "Buy to Close (Call)",
    "Sell Cash-Secured Put", "Buy to Close (Put)",
    "Assigned (Call)", "Assigned (Put)",
    "Expired Worthless", "Dividend", "Dividend (DRIP)",
    "Rotate Out", "Rotate Into",
]
_OPTION_SELL_EVENTS  = {"Sell Covered Call", "Sell Cash-Secured Put"}
_OPTION_CLOSE_EVENTS = {"Buy to Close (Call)", "Buy to Close (Put)"}
_ASSIGN_EVENTS       = {"Assigned (Call)", "Assigned (Put)"}
_OPTION_EVENTS       = _OPTION_SELL_EVENTS | _OPTION_CLOSE_EVENTS | _ASSIGN_EVENTS | {"Expired Worthless"}
_ROTATION_EVENTS     = {"Rotate Out", "Rotate Into"}

EVENT_COLORS = {
    "Buy Stock":             "#60a5fa",
    "Sell Stock":            "#a78bfa",
    "Sell Covered Call":     "#22c55e",
    "Buy to Close (Call)":   "#f87171",
    "Sell Cash-Secured Put": "#22c55e",
    "Buy to Close (Put)":    "#f87171",
    "Assigned (Call)":       "#fb923c",
    "Assigned (Put)":        "#fb923c",
    "Expired Worthless":     "#4ade80",
    "Dividend":              "#34d399",
    "Dividend (DRIP)":      "#34d399",
    "Rotate Out":            "#f59e0b",
    "Rotate Into":           "#06b6d4",
}

@st.cache_data(ttl=300)
def fetch_watchlist_prices(occ_list):
    """Fetches current option price, underlying spot price, and delta for watchlist items."""
    results = {}
    r_free = 0.045
    for occ in occ_list:
        parsed = parse_occ(occ)
        if not parsed:
            results[occ] = {'option_price': None, 'spot': None, 'delta': None, 'gamma': None}
            continue
        ticker, expiration, opt_type, strike = parsed
        try:
            underlying_ticker = _yf_ticker(ticker)
            spot, stock_date = get_latest_price(underlying_ticker)
            chain = underlying_ticker.option_chain(expiration)
            options = chain.calls if opt_type == 'C' else chain.puts
            contract = options[options['strike'] == strike]
            if contract.empty:
                results[occ] = {'option_price': None, 'spot': spot, 'delta': None, 'gamma': None}
                continue
            last_price = float(contract['lastPrice'].values[0])
            bid = float(contract['bid'].values[0]) if 'bid' in contract.columns else 0.0
            ask = float(contract['ask'].values[0]) if 'ask' in contract.columns else 0.0
            # Prefer bid/ask mid (live quotes) over lastPrice (stale for low-volume OTM options)
            if bid > 0 and ask > 0:
                option_price = (bid + ask) / 2.0
            else:
                option_price = last_price
            days_to_exp = (datetime.strptime(expiration, "%Y-%m-%d") - datetime.now()).days
            T = max(days_to_exp / 365.0, 0.001)
            # Recompute IV from the price we're actually using (bid/ask mid may differ from lastPrice)
            iv = implied_volatility(option_price, spot, strike, T, r_free, 0.0, opt_type)
            if iv is None or iv <= 0:
                iv = float(contract['impliedVolatility'].values[0])  # fall back to Yahoo's value
            delta, _, gamma = calculate_greeks(spot, strike, T, r_free, iv, opt_type)

            # Use approximation if option data timestamp is different from stock price timestamp
            if 'lastTradeDate' in contract.columns:
                trade_time = pd.to_datetime(contract['lastTradeDate'].values[0])
                if trade_time.tzinfo is None:
                    trade_time = trade_time.tz_localize('UTC')
                stock_time = stock_date
                if stock_time.tzinfo is None:
                    stock_time = stock_time.tz_localize('UTC')

                if trade_time < stock_time and iv > 0:
                    approx_price, _ = approximate_realtime_option_price(underlying_ticker, ticker, spot, strike, option_price, trade_time, expiration, opt_type)
                    if approx_price > 0:
                        option_price = approx_price
            
            results[occ] = {'option_price': option_price, 'spot': spot, 'delta': delta, 'gamma': gamma, 'iv': iv, 'strike': strike, 'T': T, 'opt_type': opt_type}
        except Exception:
            results[occ] = {'option_price': None, 'spot': None, 'delta': None, 'gamma': None, 'iv': None, 'strike': None, 'T': None, 'opt_type': None}
    return results

def load_watchlist():
    if not os.path.exists(WATCHLIST_FILE):
        return pd.DataFrame(columns=WATCHLIST_COLS)
    df = pd.read_csv(WATCHLIST_FILE)
    for col in WATCHLIST_COLS:
        if col not in df.columns:
            df[col] = '' if col in ('Label', 'Intent') else np.nan
    # Backward compat: rows without ItemType are options
    df['ItemType'] = df['ItemType'].fillna('option')
    df.loc[df['ItemType'].str.strip() == '', 'ItemType'] = 'option'
    return df

def save_watchlist(df):
    df[WATCHLIST_COLS].to_csv(WATCHLIST_FILE, index=False)

# ---- TRADES helpers ----
def load_trades():
    if not os.path.exists(TRADES_FILE):
        return pd.DataFrame(columns=TRADES_COLS)
    df = pd.read_csv(TRADES_FILE)
    for col in TRADES_COLS:
        if col not in df.columns:
            df[col] = ''
    df['date']             = pd.to_datetime(df['date'], errors='coerce')
    df['qty']              = pd.to_numeric(df['qty'],              errors='coerce').fillna(0)
    df['price']            = pd.to_numeric(df['price'],            errors='coerce').fillna(0)
    df['fees']             = pd.to_numeric(df['fees'],             errors='coerce').fillna(0)
    df['strike']           = pd.to_numeric(df['strike'],           errors='coerce')
    df['capital_reserved'] = pd.to_numeric(df['capital_reserved'], errors='coerce')
    df['leg']              = pd.to_numeric(df['leg'],              errors='coerce')
    return df.sort_values('date', ascending=False).reset_index(drop=True)

def save_trades(df):
    out = df.copy()
    out['date'] = pd.to_datetime(out['date']).dt.strftime('%Y-%m-%d')
    out[TRADES_COLS].to_csv(TRADES_FILE, index=False)

def trade_cash_flow(row):
    """Signed cash flow: positive = money received, negative = money paid out."""
    et     = str(row.get('event_type', ''))
    qty    = float(row.get('qty',   0) or 0)
    price  = float(row.get('price', 0) or 0)
    fees   = float(row.get('fees',  0) or 0)
    strike_raw = row.get('strike')
    strike = float(strike_raw) if pd.notna(strike_raw) and strike_raw != '' else 0.0

    if et == "Buy Stock":
        return -(qty * price) - fees
    elif et == "Sell Stock":
        return  (qty * price) - fees
    elif et in _OPTION_SELL_EVENTS:          # premium received
        return  (qty * 100 * price) - fees
    elif et in _OPTION_CLOSE_EVENTS:         # cost to close
        return -(qty * 100 * price) - fees
    elif et == "Assigned (Call)":            # shares called away at strike
        return  (qty * 100 * strike) - fees
    elif et == "Assigned (Put)":             # forced to buy at strike
        return -(qty * 100 * strike) - fees
    elif et == "Expired Worthless":
        return -fees
    elif et == "Dividend":
        return price - fees                  # price = total dividend amount
    elif et == "Dividend (DRIP)":
        return -fees                         # reinvested — net cash flow is zero (minus any fees)
    elif et == "Rotate Out":
        return  (qty * price) - fees         # selling shares — money in
    elif et == "Rotate Into":
        return -(qty * price) - fees         # buying shares — money out
    return 0.0

@st.cache_data(ttl=3600)
def fetch_benchmark_history(tickers_tuple, start_date_str):
    """Fetch daily Close prices for benchmark tickers from start_date to today."""
    result = {}
    for t in tickers_tuple:
        try:
            hist = yf.download(t, start=start_date_str, progress=False, auto_adjust=True)
            if not hist.empty:
                close = hist['Close'].squeeze()
                if hasattr(close, 'iloc'):
                    result[t] = close
        except Exception:
            pass
    return result

@st.cache_data(ttl=300)
def fetch_trade_live_prices(tickers_tuple):
    """Fetch current prices for open stock positions tracked in the trades journal."""
    result = {}
    for ticker in tickers_tuple:
        try:
            t = _yf_ticker(ticker)
            price, _ = get_latest_price(t)
            result[ticker] = price
        except Exception:
            result[ticker] = None
    return result

def compute_open_stock_positions(trades_df):
    """
    Walk trades chronologically and return open share positions.
    Returns dict: {ticker: {'shares': float, 'cost_basis': float}}
    cost_basis = total cash paid to acquire the shares still held.
    """
    positions = {}   # ticker -> {'shares': float, 'cost_basis': float}
    for _, row in trades_df.sort_values('date').iterrows():
        et     = str(row.get('event_type', ''))
        ticker = str(row.get('ticker', '')).upper()
        qty    = float(row.get('qty', 0) or 0)
        price  = float(row.get('price', 0) or 0)
        strike_raw = row.get('strike')
        strike = float(strike_raw) if pd.notna(strike_raw) and str(strike_raw) not in ('', 'nan') else 0.0

        pos = positions.setdefault(ticker, {'shares': 0.0, 'cost_basis': 0.0})

        if et == "Buy Stock":
            shares_added = qty
            cost_added   = qty * price
            pos['cost_basis'] += cost_added
            pos['shares']     += shares_added

        elif et == "Sell Stock":
            if pos['shares'] > 0:
                avg = pos['cost_basis'] / pos['shares']
                pos['cost_basis'] -= avg * min(qty, pos['shares'])
            pos['shares'] = max(0.0, pos['shares'] - qty)

        elif et == "Assigned (Put)":
            shares_added = qty * 100
            cost_added   = shares_added * strike
            pos['cost_basis'] += cost_added
            pos['shares']     += shares_added

        elif et == "Assigned (Call)":
            # Shares called away — reduce position
            if pos['shares'] > 0:
                avg = pos['cost_basis'] / pos['shares']
                pos['cost_basis'] -= avg * min(qty * 100, pos['shares'])
            pos['shares'] = max(0.0, pos['shares'] - qty * 100)

        elif et == "Dividend (DRIP)":
            pos['cost_basis'] += qty * price  # shares acquired via reinvestment
            pos['shares']     += qty

        elif et == "Rotate Into":
            pos['cost_basis'] += qty * price
            pos['shares']     += qty

        elif et == "Rotate Out":
            if pos['shares'] > 0:
                avg = pos['cost_basis'] / pos['shares']
                pos['cost_basis'] -= avg * min(qty, pos['shares'])
            pos['shares'] = max(0.0, pos['shares'] - qty)

    return {t: p for t, p in positions.items() if p['shares'] > 0}

def watchlist_occ(row):
    exp_str = str(int(row['ExpirationYYMMDD']))
    yy, mm, dd = exp_str[:2], exp_str[2:4], exp_str[4:]
    strike_fmt = f"{int(float(row['Strike']) * 1000):08d}"
    return f"{str(row['Ticker']).upper()}{yy}{mm}{dd}{str(row['OptionType']).upper()}{strike_fmt}"

def watchlist_card_html(row, data):
    ticker   = str(row['Ticker']).upper()
    opt_type = str(row['OptionType']).upper()
    exp_str  = str(int(row['ExpirationYYMMDD']))
    exp_fmt  = f"{exp_str[2:4]}/{exp_str[4:]}/{exp_str[:2]}"
    strike   = float(row['Strike'])
    target   = float(row['TargetPrice'])
    intent   = str(row['Intent']).strip().lower() if pd.notna(row['Intent']) and str(row['Intent']).strip() else 'buy'
    label    = str(row['Label']) if pd.notna(row['Label']) and str(row['Label']).strip() else ''

    current_price = data.get('option_price') if data else None
    spot          = data.get('spot')          if data else None
    delta         = data.get('delta')         if data else None
    gamma         = data.get('gamma')         if data else None
    iv            = data.get('iv')            if data else None
    bs_strike     = data.get('strike')        if data else None
    bs_T          = data.get('T')             if data else None
    bs_opt_type   = data.get('opt_type')      if data else None

    strike_disp = int(strike) if strike == int(strike) else strike
    target_disp = int(target) if target == int(target) else target

    # Buy: want price to drop to target. Sell: want price to rise to target.
    if current_price is not None:
        hit = current_price <= target if intent == 'buy' else current_price >= target
    else:
        hit = False

    # Stock price needed for option to reach target: solve BS(S*) = target via bisection.
    # This is exact and handles large moves correctly, unlike delta-gamma approximation.
    stock_target_str = "N/A"
    if (spot is not None and current_price is not None and iv is not None and iv > 0
            and bs_strike is not None and bs_T is not None and bs_opt_type is not None):
        try:
            r_free = 0.045
            def bs_diff(s):
                return black_scholes(s, bs_strike, bs_T, r_free, 0.0, iv, bs_opt_type) - target
            # Search range: 1% to 5x current spot
            lo, hi = spot * 0.01, spot * 5.0
            if bs_diff(lo) * bs_diff(hi) < 0:
                s_star = brentq(bs_diff, lo, hi, xtol=0.01)
                stock_target_str = f"${s_star:,.2f}"
            # else: target is outside achievable range at current IV/T — leave as N/A
        except Exception:
            pass

    border_color = '#22c55e' if hit else '#3b82f6'
    bg_color     = 'rgba(34,197,94,0.08)' if hit else 'rgba(59,130,246,0.06)'
    price_color  = '#22c55e' if hit else '#60a5fa'
    type_color   = '#F472B6' if opt_type == 'C' else '#FB923C'
    intent_label = 'Want to Buy' if intent == 'buy' else 'Want to Sell'
    intent_color = '#34D399' if intent == 'buy' else '#F87171'

    option_price_str = f"${current_price:.2f}" if current_price is not None else "N/A"
    spot_str         = f"${spot:,.2f}"          if spot is not None          else "N/A"
    badge = '<span style="background:#22c55e;color:#000;font-size:0.65em;padding:2px 6px;border-radius:4px;margin-left:6px;font-weight:bold;">TARGET HIT</span>' if hit else ''

    return f"""
<div style="border:1.5px solid {border_color};border-radius:10px;padding:14px 18px;
            background:{bg_color};">
  <div style="font-size:1.1em;font-weight:bold;color:#e2e8f0;">
    <span style="color:#60A5FA">{ticker}</span>
    <span style="color:{type_color};margin-left:4px">{opt_type}</span>
    {badge}
  </div>
  <div style="color:#A78BFA;font-size:0.82em;margin-top:2px;">{exp_fmt}</div>
  <div style="color:#34D399;font-size:0.82em;">Strike ${strike_disp}</div>
  <div style="color:{intent_color};font-size:0.75em;margin-top:3px;font-weight:600;">{intent_label}</div>
  {"<div style='color:#9CA3AF;font-size:0.75em;margin-top:2px;'>" + label + "</div>" if label else ""}
  <div style="margin-top:10px;display:flex;justify-content:space-between;align-items:baseline;">
    <div>
      <div style="color:#9CA3AF;font-size:0.68em;text-transform:uppercase;">Option Price</div>
      <div style="font-size:1.25em;font-weight:bold;color:{price_color};">{option_price_str}</div>
      <div style="color:#C084FC;font-size:0.85em;font-weight:700;">Target ${target_disp}</div>
    </div>
    <div style="text-align:right;">
      <div style="color:#9CA3AF;font-size:0.68em;text-transform:uppercase;">{ticker} Price</div>
      <div style="font-size:1.1em;font-weight:bold;color:#e2e8f0;">{spot_str}</div>
      <div style="color:#FCD34D;font-size:0.85em;font-weight:700;">Need {stock_target_str}</div>
    </div>
  </div>
</div>"""

@st.cache_data(ttl=300)
def fetch_stock_prices(tickers):
    """Fetches price, day change, and technical indicators for a tuple of stock tickers."""
    results = {}
    for ticker in tickers:
        try:
            t = _yf_ticker(ticker)
            hist = t.history(period="1y", prepost=True)
            if hist.empty:
                raise ValueError("No data returned")

            close = hist['Close']

            # Price & day change
            price, _  = get_latest_price(t)
            prev   = float(close.iloc[-2]) if len(close) >= 2 else float(close.iloc[-1])
            change     = price - prev
            change_pct = (change / prev * 100) if prev else 0

            # EMA 50 & EMA 200
            ema50  = float(close.ewm(span=50,  adjust=False).mean().iloc[-1])
            ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])

            # Bollinger Bands (50-day SMA, 2× population stdev — matches TradingView default)
            bb_window = 50
            sma50    = close.rolling(bb_window).mean()
            std50    = close.rolling(bb_window).std(ddof=0)
            bb_upper = float((sma50 + 2 * std50).iloc[-1])
            bb_lower = float((sma50 - 2 * std50).iloc[-1])

            # RSI (14-period, Wilder's smoothing)
            delta  = close.diff()
            gain   = delta.clip(lower=0)
            loss   = (-delta).clip(lower=0)
            avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
            rs  = avg_gain.iloc[-1] / avg_loss.iloc[-1] if avg_loss.iloc[-1] != 0 else float('inf')
            rsi = 100 - (100 / (1 + rs))

            results[ticker] = {
                'price': price, 'change': change, 'change_pct': change_pct,
                'ema50': ema50, 'ema200': ema200,
                'bb_upper': bb_upper, 'bb_lower': bb_lower,
                'rsi': rsi,
            }
        except Exception:
            results[ticker] = {
                'price': None, 'change': None, 'change_pct': None,
                'ema50': None, 'ema200': None,
                'bb_upper': None, 'bb_lower': None,
                'rsi': None,
            }
    return results

def stock_card_html(row, data):
    ticker = str(row['Ticker']).upper()
    target = float(row['TargetPrice']) if pd.notna(row['TargetPrice']) and float(row['TargetPrice']) != 0 else None
    intent = str(row['Intent']).strip().lower() if pd.notna(row['Intent']) and str(row['Intent']).strip() else 'buy'
    label  = str(row['Label']) if pd.notna(row['Label']) and str(row['Label']).strip() else ''

    price      = data.get('price')      if data else None
    change     = data.get('change')     if data else None
    change_pct = data.get('change_pct') if data else None
    rsi        = data.get('rsi')        if data else None
    bb_upper   = data.get('bb_upper')   if data else None
    bb_lower   = data.get('bb_lower')   if data else None
    ema50      = data.get('ema50')      if data else None
    ema200     = data.get('ema200')     if data else None

    # --- Signal detection ---
    sell_signal = (price is not None and bb_upper is not None and rsi is not None
                   and price > bb_upper and rsi > 80)
    buy_signal  = (price is not None and bb_lower is not None and rsi is not None
                   and price < bb_lower and rsi < 25)

    # --- Alert target hit ---
    hit = False
    if price is not None and target is not None:
        hit = price <= target if intent == 'buy' else price >= target

    # --- Card colors: signal takes priority over alert hit ---
    if sell_signal:
        border_color = '#f87171'
        bg_color     = 'rgba(248,113,113,0.10)'
        price_color  = '#f87171'
    elif buy_signal:
        border_color = '#22c55e'
        bg_color     = 'rgba(34,197,94,0.10)'
        price_color  = '#22c55e'
    elif hit:
        border_color = '#22c55e'
        bg_color     = 'rgba(34,197,94,0.08)'
        price_color  = '#22c55e'
    else:
        border_color = '#3b82f6'
        bg_color     = 'rgba(59,130,246,0.06)'
        price_color  = '#60a5fa'

    intent_label = 'Alert: Buy below' if intent == 'buy' else 'Alert: Sell above'
    intent_color = '#34D399' if intent == 'buy' else '#F87171'
    target_badge = '<span style="background:#22c55e;color:#000;font-size:0.65em;padding:2px 6px;border-radius:4px;margin-left:6px;font-weight:bold;">TARGET HIT</span>' if hit else ''

    price_str = f"${price:,.2f}" if price is not None else "N/A"
    if change is not None and change_pct is not None:
        chg_color = '#22c55e' if change >= 0 else '#f87171'
        chg_arrow = '▲' if change >= 0 else '▼'
        chg_str = f'<span style="color:{chg_color};font-size:0.8em;">{chg_arrow} ${abs(change):,.2f} ({change_pct:+.2f}%)</span>'
    else:
        chg_str = '<span style="color:#6B7280;font-size:0.8em;">N/A</span>'

    target_row = ''
    if target is not None:
        target_disp = int(target) if target == int(target) else target
        target_row = f'<div style="color:{intent_color};font-size:0.75em;margin-top:3px;font-weight:600;">{intent_label} ${target_disp} {target_badge}</div>'

    # Signal badge
    if sell_signal:
        signal_badge = '<span style="background:#f87171;color:#000;font-size:0.65em;padding:2px 8px;border-radius:4px;font-weight:bold;">SELL SIGNAL</span>'
    elif buy_signal:
        signal_badge = '<span style="background:#22c55e;color:#000;font-size:0.65em;padding:2px 8px;border-radius:4px;font-weight:bold;">BUY SIGNAL</span>'
    else:
        signal_badge = ''

    # RSI color
    if rsi is not None:
        rsi_color = '#f87171' if rsi > 70 else ('#22c55e' if rsi < 30 else '#e2e8f0')
        rsi_str   = f'<span style="color:{rsi_color};font-weight:700;">{rsi:.1f}</span>'
    else:
        rsi_str = '<span style="color:#6B7280;">N/A</span>'

    def fmt(v):
        return f"${v:,.2f}" if v is not None else "N/A"

    # EMA relationship hint
    ema_hint = ''
    if ema50 is not None and ema200 is not None:
        if ema50 > ema200:
            ema_hint = ' <span style="color:#22c55e;font-size:0.7em;">▲ Bull</span>'
        else:
            ema_hint = ' <span style="color:#f87171;font-size:0.7em;">▼ Bear</span>'

    indicators_html = f"""
  <div style="margin-top:10px;padding-top:8px;border-top:1px solid #2d3748;">
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 10px;font-size:0.75em;">
      <div><span style="color:#6B7280;">RSI(14)</span>&nbsp;{rsi_str}</div>
      <div><span style="color:#6B7280;">BB Upper</span>&nbsp;<span style="color:#e2e8f0;">{fmt(bb_upper)}</span></div>
      <div><span style="color:#6B7280;">EMA 50</span>&nbsp;<span style="color:#A78BFA;">{fmt(ema50)}</span>{ema_hint}</div>
      <div><span style="color:#6B7280;">BB Lower</span>&nbsp;<span style="color:#e2e8f0;">{fmt(bb_lower)}</span></div>
      <div><span style="color:#6B7280;">EMA 200</span>&nbsp;<span style="color:#A78BFA;">{fmt(ema200)}</span></div>
    </div>
  </div>"""

    return f"""
<div style="border:1.5px solid {border_color};border-radius:10px;padding:14px 18px;
            background:{bg_color};">
  <div style="display:flex;align-items:center;justify-content:space-between;">
    <div style="font-size:1.1em;font-weight:bold;">
      <span style="color:#60A5FA">{ticker}</span>
      <span style="color:#6B7280;font-size:0.7em;margin-left:6px;">STOCK</span>
    </div>
    {signal_badge}
  </div>
  {"<div style='color:#9CA3AF;font-size:0.75em;margin-top:2px;'>" + label + "</div>" if label else ""}
  <div style="margin-top:8px;">
    <div style="color:#9CA3AF;font-size:0.68em;text-transform:uppercase;">Price</div>
    <div style="font-size:1.35em;font-weight:bold;color:{price_color};line-height:1.2;">{price_str}</div>
    <div style="margin-top:2px;">{chg_str}</div>
    {target_row}
  </div>
  {indicators_html}
</div>"""

def _pdf_option_to_occ(name):
    """Convert PDF option name like 'ORCL 04/17/2026 Put $155.00' to OCC symbol string."""
    m = re.search(r'(\w+)\s+(\d{2})/(\d{2})/(\d{4})\s+(Call|Put)\s+\$(\d+(?:\.\d+)?)', name)
    if not m:
        return None
    ticker, mm, dd, yyyy, cp, strike_str = m.groups()
    yy = yyyy[2:]
    cp_char = 'C' if cp == 'Call' else 'P'
    strike_int = int(float(strike_str) * 1000)
    return f"{ticker}{yy}{mm}{dd}{cp_char}{strike_int:08d}"

def _format_pdf_symbol(name, symbol, is_option):
    """Format display symbol: equity → ticker, option → 'ORCL P 041726 $155'."""
    if not is_option:
        return symbol
    m = re.search(r'(\w+)\s+(\d{2})/(\d{2})/(\d{4})\s+(Call|Put)\s+\$(\d+(?:\.\d+)?)', name)
    if not m:
        return f"{symbol} (opt)"
    ticker, mm, dd, yyyy, cp, strike_str = m.groups()
    yy = yyyy[2:]
    cp_char = 'C' if cp == 'Call' else 'P'
    strike = float(strike_str)
    strike_fmt = int(strike) if strike == int(strike) else strike
    return f"{ticker} {cp_char} {mm}{dd}{yy} ${strike_fmt}"

@st.cache_data(ttl=300)
def fetch_summary_equity_prices(tickers_tuple):
    """Lightweight batch equity price fetch for the Summary tab (5-min cache)."""
    result = {}
    for ticker in tickers_tuple:
        try:
            t = _yf_ticker(ticker)
            price, _ = get_latest_price(t)
            result[ticker] = price
        except Exception:
            result[ticker] = None
    return result

with page_tab4:
    st.markdown('<div style="font-size:1em;font-weight:700;color:#e2e8f0;padding:2px 0 10px;">Portfolio Summary</div>', unsafe_allow_html=True)

    try:
        pdf_accounts = parse_portfolio_pdfs()
    except Exception as _e:
        st.error(f"Error parsing portfolio PDFs: {_e}")
        pdf_accounts = {}

    if not pdf_accounts:
        st.warning("No portfolio PDFs found in positions/portfolio/. Drop Robinhood statement PDFs there and refresh.")
    else:
        acct_keys = sorted(pdf_accounts.keys())
        sum_sel = st.multiselect(
            "Filter by Account",
            options=["All"] + acct_keys,
            default=["All"],
            key="sum_acct_filter"
        )
        selected_accts = acct_keys if ("All" in sum_sel or not sum_sel) else [k for k in sum_sel if k in pdf_accounts]

        # --- TOP METRICS (aggregated across selected accounts) ---
        total_pv  = sum(pdf_accounts[k]['portfolio_value']   or 0 for k in selected_accts)
        total_sec = sum(pdf_accounts[k]['total_securities']  or 0 for k in selected_accts)
        total_div = sum(pdf_accounts[k]['dividends_period']  or 0 for k in selected_accts)
        period_str = pdf_accounts[selected_accts[0]]['period'] or "" if selected_accts else ""

        st.markdown(f"""
<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px;">
  <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:14px 20px;flex:2;min-width:180px;">
    <div style="color:#9CA3AF;font-size:0.65em;text-transform:uppercase;margin-bottom:4px;">Total Portfolio Value</div>
    <div style="font-size:1.5em;font-weight:700;color:#e2e8f0;">${total_pv:,.2f}</div>
    <div style="color:#6B7280;font-size:0.7em;margin-top:3px;">{period_str}</div>
  </div>
  <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:14px 20px;flex:1;min-width:140px;">
    <div style="color:#9CA3AF;font-size:0.65em;text-transform:uppercase;margin-bottom:4px;">Total Securities</div>
    <div style="font-size:1.5em;font-weight:700;color:#e2e8f0;">${total_sec:,.2f}</div>
  </div>
  <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:14px 20px;flex:1;min-width:140px;">
    <div style="color:#9CA3AF;font-size:0.65em;text-transform:uppercase;margin-bottom:4px;">Dividends (Period)</div>
    <div style="font-size:1.5em;font-weight:700;color:#22c55e;">${total_div:,.2f}</div>
  </div>
  <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:14px 20px;flex:1;min-width:100px;">
    <div style="color:#9CA3AF;font-size:0.65em;text-transform:uppercase;margin-bottom:4px;">Accounts</div>
    <div style="font-size:1.5em;font-weight:700;color:#e2e8f0;">{len(selected_accts)}</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # --- PER-ACCOUNT SUMMARY CARDS ---
        cards_row = '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px;">'
        for key in selected_accts:
            acct = pdf_accounts[key]
            pv   = acct['portfolio_value']  or 0
            cash = acct['cash_balance']     or 0
            sec  = acct['total_securities'] or 0
            div  = acct['dividends_period'] or 0
            cash_color = '#22c55e' if cash >= 0 else '#f87171'
            cards_row += f"""
<div style="border:1px solid rgba(255,255,255,0.12);border-radius:10px;padding:12px 16px;flex:1;min-width:200px;background:rgba(255,255,255,0.02);">
  <div style="font-size:0.85em;font-weight:700;color:#e2e8f0;">{key}</div>
  <div style="color:#6B7280;font-size:0.65em;margin-bottom:8px;">{acct['type']} &bull; {acct.get('period','')}</div>
  <div style="font-size:1.2em;font-weight:700;color:#e2e8f0;margin-bottom:6px;">${pv:,.2f}</div>
  <div style="display:flex;gap:16px;flex-wrap:wrap;">
    <div><div style="color:#6B7280;font-size:0.6em;text-transform:uppercase;">Securities</div><div style="color:#e2e8f0;font-size:0.85em;font-weight:600;">${sec:,.2f}</div></div>
    <div><div style="color:#6B7280;font-size:0.6em;text-transform:uppercase;">Cash</div><div style="color:{cash_color};font-size:0.85em;font-weight:600;">${cash:,.2f}</div></div>
    <div><div style="color:#6B7280;font-size:0.6em;text-transform:uppercase;">Dividends</div><div style="color:#22c55e;font-size:0.85em;font-weight:600;">${div:,.2f}</div></div>
  </div>
</div>"""
        cards_row += '</div>'
        st.markdown(cards_row, unsafe_allow_html=True)

        # --- AGGREGATE HOLDINGS ACROSS SELECTED ACCOUNTS ---
        from collections import defaultdict
        aggregated = defaultdict(lambda: {
            'display_symbol': '', 'is_option': False, 'is_short': False,
            'sum_qty': 0.0, 'sum_cost_basis': 0.0, 'has_cost_basis': False,
            'occ': None, 'ticker': None,
        })
        for key in selected_accts:
            for h in pdf_accounts[key]['holdings']:
                disp = _format_pdf_symbol(h['name'], h['symbol'], h['is_option'])
                row = aggregated[disp]
                row['display_symbol'] = disp
                row['is_option'] = h['is_option']
                row['is_short'] = h['is_short']
                row['sum_qty'] += h['qty']
                if h.get('cost_basis') is not None:
                    row['sum_cost_basis'] += h['cost_basis']
                    row['has_cost_basis'] = True
                if h['is_option'] and row['occ'] is None:
                    row['occ'] = _pdf_option_to_occ(h['name'])
                if not h['is_option']:
                    row['ticker'] = h['symbol']

        # Fetch current prices
        equity_tickers = tuple(sorted(set(v['ticker'] for v in aggregated.values() if v['ticker'])))
        opt_occs = tuple(sorted(set(v['occ'] for v in aggregated.values() if v['occ'])))

        eq_prices = {}
        opt_price_data = {}
        if equity_tickers:
            with st.spinner("Fetching current equity prices…"):
                eq_prices = fetch_summary_equity_prices(equity_tickers)
        if opt_occs:
            with st.spinner("Fetching current option prices…"):
                opt_price_data = fetch_watchlist_prices(opt_occs)

        # Build display rows
        table_rows = []
        for disp, row in aggregated.items():
            qty = row['sum_qty']
            cost_basis = row['sum_cost_basis'] if row['has_cost_basis'] else None
            avg_cost = (cost_basis / abs(qty)) if (cost_basis and qty) else None

            if row['is_option']:
                occ = row['occ']
                opt_d = opt_price_data.get(occ, {}) if occ else {}
                curr_price = opt_d.get('option_price')
            else:
                curr_price = eq_prices.get(row['ticker'])

            curr_value = (curr_price * qty) if curr_price is not None else None
            gain_loss  = (curr_value - cost_basis) if (curr_value is not None and cost_basis is not None) else None
            pct_gl     = (gain_loss / abs(cost_basis) * 100) if (gain_loss is not None and cost_basis) else None

            table_rows.append({
                '_symbol': disp,
                '_is_option': row['is_option'],
                '_curr_value_raw': curr_value,
                'Symbol': disp,
                'Sum Qty': qty,
                'Cost Basis': cost_basis,
                'Avg Cost': avg_cost,
                'Curr Price': curr_price,
                'Curr Value': curr_value,
                'Gain/Loss': gain_loss,
                '% G/L': pct_gl,
                'Alloc%': None,  # filled below
            })

        total_curr_value = sum(r['_curr_value_raw'] or 0 for r in table_rows)
        for r in table_rows:
            cv = r['_curr_value_raw']
            r['Alloc%'] = (cv / total_curr_value * 100) if (cv and total_curr_value) else None

        def _make_holdings_df(rows):
            df = pd.DataFrame(rows, columns=[
                'Symbol', 'Sum Qty', 'Cost Basis', 'Avg Cost',
                'Curr Price', 'Curr Value', 'Gain/Loss', '% G/L', 'Alloc%'
            ])
            df = df.sort_values('Curr Value', key=lambda x: pd.to_numeric(x, errors='coerce').abs().fillna(0), ascending=False)

            def fmt_dollar(v):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return '—'
                return f"${v:,.2f}"

            def fmt_pct(v):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return '—'
                sign = '+' if v > 0 else ''
                return f"{sign}{v:.2f}%"

            def fmt_qty(v):
                if v is None:
                    return '—'
                return f"{v:,.4g}"

            def fmt_gl(v):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return '—'
                sign = '+' if v > 0 else ''
                return f"{sign}${v:,.2f}"

            df['Sum Qty']    = df['Sum Qty'].map(fmt_qty)
            df['Cost Basis'] = df['Cost Basis'].map(fmt_dollar)
            df['Avg Cost']   = df['Avg Cost'].map(fmt_dollar)
            df['Curr Price'] = df['Curr Price'].map(fmt_dollar)
            df['Curr Value'] = df['Curr Value'].map(fmt_dollar)
            df['Gain/Loss']  = df['Gain/Loss'].map(fmt_gl)
            df['% G/L']      = df['% G/L'].map(fmt_pct)
            df['Alloc%']     = df['Alloc%'].map(lambda v: f"{v:.2f}%" if v is not None and not (isinstance(v, float) and pd.isna(v)) else '—')
            return df

        eq_rows  = [r for r in table_rows if not r['_is_option']]
        opt_rows = [r for r in table_rows if r['_is_option']]

        col_r, col_r2, _ = st.columns([1, 1, 3])
        if col_r.button("Refresh Prices", key="sum_refresh"):
            fetch_summary_equity_prices.clear()
            fetch_watchlist_prices.clear()
            st.rerun()
        if col_r2.button("Reload PDFs", key="sum_reload_pdf"):
            parse_portfolio_pdfs.clear()
            st.rerun()

        if eq_rows:
            with st.expander(f"Equities & ETFs ({len(eq_rows)})", expanded=True):
                st.dataframe(_make_holdings_df(eq_rows), use_container_width=True, hide_index=True)
        if opt_rows:
            with st.expander(f"Options ({len(opt_rows)})", expanded=False):
                st.dataframe(_make_holdings_df(opt_rows), use_container_width=True, hide_index=True)

with page_tab2:
    watchlist = load_watchlist()

    # Live mode toggle (off by default, resets when page is closed)
    if 'wl_live' not in st.session_state:
        st.session_state['wl_live'] = False
    if 'wl_interval' not in st.session_state:
        st.session_state['wl_interval'] = 30

    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 4])
    live_on = ctrl1.toggle("Live Quotes", value=st.session_state['wl_live'], key="wl_live_toggle")
    st.session_state['wl_live'] = live_on

    if live_on:
        interval = ctrl2.selectbox("Refresh every", [15, 30, 60, 120], index=1, format_func=lambda x: f"{x}s", key="wl_interval_sel")
        st.session_state['wl_interval'] = interval

    if not watchlist.empty:
        option_rows = watchlist[watchlist['ItemType'] == 'option']
        stock_rows  = watchlist[watchlist['ItemType'] == 'stock']

        option_occs = {idx: watchlist_occ(row) for idx, row in option_rows.iterrows()}
        stock_tickers = tuple(stock_rows['Ticker'].str.upper().unique()) if not stock_rows.empty else ()

        # Live mode bypasses cache; manual mode uses 2-min cache
        if live_on:
            fetch_watchlist_prices.clear()
            fetch_stock_prices.clear()

        option_prices = {}
        if not option_rows.empty:
            with st.spinner("Fetching option prices..."):
                option_prices = fetch_watchlist_prices(tuple(option_occs.values()))

        stock_price_data = {}
        if stock_tickers:
            with st.spinner("Fetching stock prices..."):
                stock_price_data = fetch_stock_prices(stock_tickers)

        # Last updated timestamp
        updated_str = datetime.now().strftime("%H:%M:%S")
        status_color = '#22c55e' if live_on else '#6B7280'
        live_badge = (
            f'<span style="background:{status_color};color:#000;font-size:0.65em;'
            f'padding:2px 8px;border-radius:10px;font-weight:700;">{"LIVE" if live_on else "DELAYED"}</span>'
            f'&nbsp;<span style="color:#6B7280;font-size:0.72em;">Updated {updated_str}</span>'
        )
        st.markdown(live_badge, unsafe_allow_html=True)

        cards_html = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:8px;width:100%;">'
        for idx, row in watchlist.iterrows():
            if row['ItemType'] == 'stock':
                ticker = str(row['Ticker']).upper()
                cards_html += stock_card_html(row, stock_price_data.get(ticker))
            else:
                occ = option_occs.get(idx)
                cards_html += watchlist_card_html(row, option_prices.get(occ))
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 5])
        if col_a.button("Refresh", key="wl_refresh"):
            fetch_watchlist_prices.clear()
            fetch_stock_prices.clear()
            st.rerun()

        # Auto-refresh loop when live mode is on
        if live_on:
            countdown = st.empty()
            for i in range(st.session_state['wl_interval'], 0, -1):
                countdown.markdown(
                    f'<span style="color:#6B7280;font-size:0.75em;">Next refresh in {i}s</span>',
                    unsafe_allow_html=True
                )
                time.sleep(1)
            countdown.empty()
            fetch_watchlist_prices.clear()
            fetch_stock_prices.clear()
            st.rerun()
    else:
        st.info("No items in your watchlist. Add one below.")

    st.divider()
    with st.expander("Manage Watchlist", expanded=watchlist.empty):
        wl_add, wl_edit, wl_delete = st.tabs(["Add", "Edit", "Delete"])

        with wl_add:
            wl_add_kind = st.radio("Add", ["Option", "Stock"], horizontal=True, key="wl_add_kind")

            if wl_add_kind == "Option":
                with st.form("wl_add_option_form"):
                    c1, c2, c3, c4, c5, c6 = st.columns(6)
                    wl_ticker  = c1.text_input("Ticker", placeholder="NVDA").strip().upper()
                    wl_exp     = c2.text_input("Expiry YYMMDD", placeholder="270115")
                    wl_type    = c3.selectbox("Type", ["P", "C"])
                    wl_strike  = c4.number_input("Strike", min_value=0.01, step=0.5, value=100.0)
                    wl_target  = c5.number_input("Target Price", min_value=0.0, step=0.01, value=0.0)
                    wl_intent  = c6.selectbox("Intent", ["Buy", "Sell"])
                    wl_label   = st.text_input("Label (optional)", placeholder="e.g. earnings play")

                    if st.form_submit_button("Add Option"):
                        if not wl_ticker or not wl_exp:
                            st.error("Ticker and Expiry are required.")
                        else:
                            new_wl = pd.DataFrame([{
                                'Ticker': wl_ticker, 'ExpirationYYMMDD': int(wl_exp),
                                'OptionType': wl_type, 'Strike': float(wl_strike),
                                'TargetPrice': float(wl_target),
                                'Intent': wl_intent.lower(),
                                'Label': wl_label.strip(),
                                'ItemType': 'option'
                            }])
                            save_watchlist(pd.concat([watchlist, new_wl], ignore_index=True))
                            fetch_watchlist_prices.clear()
                            st.success(f"Added {wl_ticker} {wl_type} ${wl_strike} to watchlist.")
                            st.rerun()
            else:
                with st.form("wl_add_stock_form"):
                    c1, c2, c3 = st.columns([2, 2, 2])
                    wl_s_ticker = c1.text_input("Ticker", placeholder="MSFT").strip().upper()
                    wl_s_target = c2.number_input("Alert Price (optional)", min_value=0.0, step=0.01, value=0.0)
                    wl_s_intent = c3.selectbox("Alert Direction", ["Buy below", "Sell above"])
                    wl_s_label  = st.text_input("Label (optional)", placeholder="e.g. support level")

                    if st.form_submit_button("Add Stock"):
                        if not wl_s_ticker:
                            st.error("Ticker is required.")
                        else:
                            new_wl = pd.DataFrame([{
                                'Ticker': wl_s_ticker,
                                'ExpirationYYMMDD': np.nan,
                                'OptionType': np.nan,
                                'Strike': np.nan,
                                'TargetPrice': float(wl_s_target) if wl_s_target else np.nan,
                                'Intent': 'buy' if wl_s_intent == 'Buy below' else 'sell',
                                'Label': wl_s_label.strip(),
                                'ItemType': 'stock'
                            }])
                            save_watchlist(pd.concat([watchlist, new_wl], ignore_index=True))
                            fetch_stock_prices.clear()
                            st.success(f"Added {wl_s_ticker} to watchlist.")
                            st.rerun()

        with wl_edit:
            if watchlist.empty:
                st.info("No items to edit.")
            else:
                def _wl_label(r):
                    if str(r.get('ItemType', 'option')) == 'stock':
                        return f"{r['Ticker']} (Stock)"
                    return f"{r['Ticker']} {r['OptionType']} {r['ExpirationYYMMDD']} ${r['Strike']}"
                wl_labels  = [_wl_label(r) for _, r in watchlist.iterrows()]
                wl_sel     = st.selectbox("Select item", wl_labels, key="wl_edit_sel")
                wl_sel_idx = wl_labels.index(wl_sel)
                wl_row     = watchlist.iloc[wl_sel_idx]
                wl_row_type = str(wl_row.get('ItemType', 'option'))

                if wl_row_type == 'stock':
                    with st.form("wl_edit_stock_form"):
                        c1, c2, c3 = st.columns([2, 2, 2])
                        wl_e_ticker = c1.text_input("Ticker", value=str(wl_row['Ticker'])).strip().upper()
                        cur_target  = float(wl_row['TargetPrice']) if pd.notna(wl_row['TargetPrice']) else 0.0
                        wl_e_target = c2.number_input("Alert Price", value=cur_target, step=0.01)
                        cur_intent  = str(wl_row['Intent']).strip().lower() if pd.notna(wl_row['Intent']) else 'buy'
                        wl_e_intent = c3.selectbox("Alert Direction", ["Buy below", "Sell above"], index=0 if cur_intent == 'buy' else 1)
                        wl_e_label  = st.text_input("Label", value=str(wl_row['Label']) if pd.notna(wl_row['Label']) else '')
                        if st.form_submit_button("Save Changes"):
                            watchlist.at[wl_sel_idx, 'Ticker']      = wl_e_ticker
                            watchlist.at[wl_sel_idx, 'TargetPrice'] = float(wl_e_target)
                            watchlist.at[wl_sel_idx, 'Intent']      = 'buy' if wl_e_intent == 'Buy below' else 'sell'
                            watchlist.at[wl_sel_idx, 'Label']       = wl_e_label.strip()
                            save_watchlist(watchlist)
                            fetch_stock_prices.clear()
                            st.success("Watchlist item updated.")
                            st.rerun()
                else:
                    with st.form("wl_edit_option_form"):
                        c1, c2, c3, c4, c5, c6 = st.columns(6)
                        wl_e_ticker  = c1.text_input("Ticker", value=str(wl_row['Ticker'])).strip().upper()
                        wl_e_exp     = c2.text_input("Expiry YYMMDD", value=str(int(wl_row['ExpirationYYMMDD'])))
                        wl_e_type    = c3.selectbox("Type", ["P", "C"], index=0 if wl_row['OptionType'] == 'P' else 1)
                        wl_e_strike  = c4.number_input("Strike", value=float(wl_row['Strike']), step=0.5)
                        wl_e_target  = c5.number_input("Target Price", value=float(wl_row['TargetPrice']), step=0.01)
                        cur_intent   = str(wl_row['Intent']).strip().lower() if pd.notna(wl_row['Intent']) and str(wl_row['Intent']).strip() else 'buy'
                        wl_e_intent  = c6.selectbox("Intent", ["Buy", "Sell"], index=0 if cur_intent == 'buy' else 1)
                        wl_e_label   = st.text_input("Label", value=str(wl_row['Label']) if pd.notna(wl_row['Label']) else '')
                        if st.form_submit_button("Save Changes"):
                            watchlist.at[wl_sel_idx, 'Ticker']           = wl_e_ticker
                            watchlist.at[wl_sel_idx, 'ExpirationYYMMDD'] = int(wl_e_exp)
                            watchlist.at[wl_sel_idx, 'OptionType']       = wl_e_type
                            watchlist.at[wl_sel_idx, 'Strike']           = float(wl_e_strike)
                            watchlist.at[wl_sel_idx, 'TargetPrice']      = float(wl_e_target)
                            watchlist.at[wl_sel_idx, 'Intent']           = wl_e_intent.lower()
                            watchlist.at[wl_sel_idx, 'Label']            = wl_e_label.strip()
                            save_watchlist(watchlist)
                            fetch_watchlist_prices.clear()
                            st.success("Watchlist item updated.")
                            st.rerun()

        with wl_delete:
            if watchlist.empty:
                st.info("No items to delete.")
            else:
                def _wl_del_label(r):
                    if str(r.get('ItemType', 'option')) == 'stock':
                        return f"{r['Ticker']} (Stock)"
                    return f"{r['Ticker']} {r['OptionType']} {r['ExpirationYYMMDD']} ${r['Strike']}"
                wl_del_labels = [_wl_del_label(r) for _, r in watchlist.iterrows()]
                wl_del_sel    = st.selectbox("Select item to remove", wl_del_labels, key="wl_del_sel")
                wl_del_idx    = wl_del_labels.index(wl_del_sel)

                st.warning(f"Remove **{wl_del_sel}** from watchlist?")
                if st.button("Remove", type="primary", key="wl_del_btn"):
                    save_watchlist(watchlist.drop(index=wl_del_idx).reset_index(drop=True))
                    fetch_watchlist_prices.clear()
                    fetch_stock_prices.clear()
                    st.success("Removed from watchlist.")
                    st.rerun()
# =====================================================================
# --- SENTIMENT TAB ---
# =====================================================================

@st.cache_data(ttl=300)
def fetch_sentiment_prices(tickers):
    """Fetches current prices for a list of tickers from Yahoo Finance."""
    results = {}
    for ticker_id, ticker_info in tickers.items():
        try:
            ticker = _yf_ticker(ticker_id)
            hist = ticker.history(period="5d", prepost=True)
            price, _ = get_latest_price(ticker)
            prev  = float(hist['Close'].iloc[-2]) if len(hist) >= 2 else float(hist['Close'].iloc[-1])
            change = price - prev
            change_pct = (change / prev * 100) if prev else 0
            results[ticker_id] = {
                "name": ticker_info["name"],
                "price": price,
                "change": change,
                "change_pct": change_pct,
                "color": ticker_info["color"],
                "unit": ticker_info.get("unit", "$"),
            }
        except Exception as e:
            results[ticker_id] = {"error": str(e), "name": ticker_info["name"]}
    return results

@st.cache_data(ttl=300)
def fetch_ibit_approx():
    """Estimate current IBIT price using BTC-USD ratio (useful when market is closed)."""
    try:
        ibit = _yf_ticker("IBIT")
        btc = _yf_ticker("BTC-USD")

        ibit_hist = ibit.history(period="5d", interval="1d")
        btc_hist = btc.history(period="5d", interval="1d")

        if ibit_hist.empty or btc_hist.empty:
            return {"error": "No data", "name": "IBIT ~", "approx": True}

        ibit_close = float(ibit_hist['Close'].iloc[-1])

        # Match BTC close to last IBIT trading day
        ibit_last_date = ibit_hist.index[-1].date()
        btc_same_day = btc_hist[[d.date() == ibit_last_date for d in btc_hist.index]]
        btc_close = float(btc_same_day['Close'].iloc[-1]) if not btc_same_day.empty else float(btc_hist['Close'].iloc[-1])

        ratio = ibit_close / btc_close  # IBIT price per $1 of BTC

        btc_current, _ = get_latest_price(btc)
        approx_price = ratio * btc_current
        change = approx_price - ibit_close
        change_pct = (change / ibit_close * 100) if ibit_close else 0

        return {
            "name": "IBIT ~",
            "price": approx_price,
            "change": change,
            "change_pct": change_pct,
            "color": "#F7931A",
            "approx": True,
        }
    except Exception as e:
        return {"error": str(e), "name": "IBIT ~", "approx": True}

@st.cache_data(ttl=1800)  # cache 30 min
def fetch_cnn_fear_greed():
    """Fetches CNN Fear & Greed index via their internal data endpoint."""
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.cnn.com/markets/fear-and-greed",
        "Origin": "https://www.cnn.com",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        fg = data["fear_and_greed"]
        return {
            "score":      round(fg["score"], 1),
            "rating":     fg["rating"].replace("_", " ").title(),
            "prev_close": round(fg["previous_close"], 1),
            "prev_week":  round(fg["previous_1_week"], 1),
            "prev_month": round(fg["previous_1_month"], 1),
        }
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=1800)
def fetch_crypto_fear_greed():
    """Fetches Crypto Fear & Greed index from alternative.me public API."""
    url = "https://api.alternative.me/fng/?limit=30&format=json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        entries = data["data"]
        latest = entries[0]
        return {
            "score":      int(latest["value"]),
            "rating":     latest["value_classification"],
            "yesterday":  int(entries[1]["value"]),
            "prev_week":  int(entries[6]["value"]),
            "prev_month": int(entries[29]["value"]),
            "history":    [(int(e["value"]), e["value_classification"]) for e in reversed(entries)],
        }
    except Exception as e:
        return {"error": str(e)}

def sentiment_color(score):
    if score <= 25:   return "#ef4444", "Extreme Fear"
    if score <= 44:   return "#f97316", "Fear"
    if score <= 55:   return "#eab308", "Neutral"
    if score <= 74:   return "#84cc16", "Greed"
    return                   "#22c55e", "Extreme Greed"

def gauge_html(score, label, title, sub_rows):
    color, _ = sentiment_color(score)
    # Arc: 180° sweep, score maps 0-100 to 0-180 degrees
    angle   = score * 1.8 - 90   # -90 = leftmost, +90 = rightmost
    rad     = angle * 3.14159 / 180
    nx      = 100 + 75 * __import__('math').cos(rad)
    ny      = 100 - 75 * __import__('math').sin(rad)

    sub_html = ""
    for lbl, val in sub_rows:
        sc = val if isinstance(val, (int, float)) else 0
        c, _ = sentiment_color(sc)
        sub_html += (
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.05);">'
            f'<span style="color:#9CA3AF;font-size:0.75em;">{lbl}</span>'
            f'<span style="color:{c};font-weight:600;font-size:0.8em;">{val} — {_}</span>'
            f'</div>'
        )
        _, _ = sentiment_color(sc)

    # Re-do sub_html properly (need the rating text per row)
    sub_html = ""
    for lbl, val in sub_rows:
        sc = val if isinstance(val, (int, float)) else 0
        c, rating_txt = sentiment_color(sc)
        sub_html += (
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">'
            f'<span style="color:#9CA3AF;font-size:0.75em;">{lbl}</span>'
            f'<span style="color:{c};font-weight:600;font-size:0.8em;">{val} — {rating_txt}</span>'
            f'</div>'
        )

    return f"""
<div style="border:1.5px solid {color};border-radius:14px;padding:20px 24px;
            background:rgba(255,255,255,0.03);max-width:340px;display:inline-block;
            vertical-align:top;margin:8px;">
  <div style="color:#9CA3AF;font-size:0.75em;text-transform:uppercase;
              letter-spacing:0.08em;margin-bottom:12px;">{title}</div>
  <svg viewBox="0 0 200 110" width="200" height="110" style="display:block;margin:0 auto 6px;">
    <!-- background arc -->
    <path d="M 25 100 A 75 75 0 0 1 175 100" fill="none" stroke="#374151" stroke-width="14" stroke-linecap="round"/>
    <!-- colored arc -->
    <path d="M 25 100 A 75 75 0 0 1 175 100" fill="none" stroke="#374151" stroke-width="14"
          stroke-dasharray="235.6" stroke-dashoffset="{235.6 * (1 - score/100):.1f}"
          stroke-linecap="round" stroke="#374151"/>
    <!-- gradient arc segments -->
    <path d="M 25 100 A 75 75 0 0 1 68 28"  fill="none" stroke="#ef4444" stroke-width="14" stroke-linecap="butt"/>
    <path d="M 68 28  A 75 75 0 0 1 100 25" fill="none" stroke="#f97316" stroke-width="14" stroke-linecap="butt"/>
    <path d="M 100 25 A 75 75 0 0 1 132 28" fill="none" stroke="#eab308" stroke-width="14" stroke-linecap="butt"/>
    <path d="M 132 28 A 75 75 0 0 1 175 100" fill="none" stroke="#22c55e" stroke-width="14" stroke-linecap="butt"/>
    <!-- needle -->
    <line x1="100" y1="100" x2="{nx:.1f}" y2="{ny:.1f}"
          stroke="white" stroke-width="2.5" stroke-linecap="round"/>
    <circle cx="100" cy="100" r="5" fill="white"/>
    <!-- score label -->
    <text x="100" y="92" text-anchor="middle" fill="{color}"
          font-size="22" font-weight="bold">{score}</text>
  </svg>
  <div style="text-align:center;color:{color};font-size:1.1em;font-weight:700;
              margin-bottom:14px;">{label}</div>
  <div>{sub_html}</div>
</div>"""

with page_tab3:
    st.markdown('<div style="font-size:1em;font-weight:700;color:#e2e8f0;padding:2px 0 8px;">Market Sentiment</div>', unsafe_allow_html=True)

    # Live mode toggle for Sentiment
    if 'sent_live' not in st.session_state:
        st.session_state['sent_live'] = False
    if 'sent_interval' not in st.session_state:
        st.session_state['sent_interval'] = 30

    sent_ctrl1, sent_ctrl2, sent_ctrl3 = st.columns([2, 2, 4])
    sent_live_on = sent_ctrl1.toggle("Live Quotes", value=st.session_state['sent_live'], key="sent_live_toggle")
    st.session_state['sent_live'] = sent_live_on
    if sent_live_on:
        sent_interval = sent_ctrl2.selectbox("Refresh every", [15, 30, 60, 120], index=1, format_func=lambda x: f"{x}s", key="sent_interval_sel")
        st.session_state['sent_interval'] = sent_interval

    if sent_live_on:
        fetch_sentiment_prices.clear()

    # Prices
    tickers_to_fetch = {
        "BTC-USD": {"name": "₿ BTC", "color": "#F7931A"},
        "QQQ": {"name": "QQQ", "color": "#A78BFA"},
        "SPY": {"name": "SPY", "color": "#A78BFA"},
        "CL=F": {"name": "NY Crude", "color": "#FB923C"},
        "BZ=F": {"name": "Brent", "color": "#FB923C"},
        "ES=F": {"name": "S&P 500 F", "color": "#34D399"},
        "NQ=F": {"name": "Nasdaq 100 F", "color": "#34D399"},
        "^TNX": {"name": "10yr Yield", "color": "#FCD34D", "unit": "%"},
    }
    prices = fetch_sentiment_prices(tickers_to_fetch)

    # Inject approximate IBIT price derived from live BTC
    if sent_live_on:
        fetch_ibit_approx.clear()
    ibit_approx = fetch_ibit_approx()
    prices["IBIT"] = ibit_approx

    sent_updated_str = datetime.now().strftime("%H:%M:%S")
    sent_status_color = '#22c55e' if sent_live_on else '#6B7280'
    sent_live_badge = (
        f'<span style="background:{sent_status_color};color:#000;font-size:0.65em;'
        f'padding:2px 8px;border-radius:10px;font-weight:700;">{"LIVE" if sent_live_on else "DELAYED"}</span>'
        f'&nbsp;<span style="color:#6B7280;font-size:0.72em;">Updated {sent_updated_str}</span>'
    )

    prices_html = '<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:12px;">'
    for ticker_id, data in prices.items():
        is_approx = data.get("approx", False)
        if "error" in data:
            price_html = f'<span style="color:#f87171;font-size:0.9em;">Error</span>'
        else:
            change_color = '#22c55e' if data['change'] >= 0 else '#f87171'
            arrow = '▲' if data['change'] >= 0 else '▼'
            approx_badge = (
                '<span style="color:#6B7280;font-size:0.65em;font-weight:500;'
                'background:#2d3748;padding:1px 5px;border-radius:4px;margin-left:4px;">est</span>'
                if is_approx else ''
            )
            unit = data.get("unit", "$")
            if unit == "%":
                price_fmt = f'{data["price"]:.2f}%'
                change_fmt = f'{arrow} {abs(data["change"]) * 100:.1f}bp ({data["change_pct"]:+.2f}%)'
            else:
                price_fmt = f'${data["price"]:,.2f}'
                change_fmt = f'{arrow} {abs(data["change"]):,.2f} ({data["change_pct"]:+.2f}%)'
            price_html = (
                f'<span style="color:#e2e8f0;font-size:1.15em;font-weight:700;">{price_fmt}</span>'
                f'{approx_badge}'
                f'<span style="color:{change_color};font-size:0.85em;font-weight:600;margin-left:8px;">'
                f'{change_fmt}</span>'
            )
        prices_html += (
            f'<div style="display:flex;align-items:center;gap:8px;padding:8px 12px;'
            f'background:#1a1f2e;border-radius:8px;border:1px solid #2d3748;">'
            f'<span style="color:{data["color"]};font-size:1em;font-weight:700;">{data["name"]}</span>'
            f'{price_html}'
            f'</div>'
        )
    prices_html += f'<div style="margin-left:8px;">{sent_live_badge}</div></div>'
    st.markdown(prices_html, unsafe_allow_html=True)


    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### CNN Fear & Greed")
        cnn = fetch_cnn_fear_greed()
        if "error" in cnn:
            st.error(f"Could not fetch CNN data: {cnn['error']}")
        else:
            sub = [
                ("Previous Close", cnn["prev_close"]),
                ("1 Week Ago",     cnn["prev_week"]),
                ("1 Month Ago",    cnn["prev_month"]),
            ]
            st.markdown(
                gauge_html(cnn["score"], f"{cnn['score']} — {cnn['rating']}", "CNN Fear & Greed Index", sub),
                unsafe_allow_html=True
            )

    with col2:
        st.markdown("#### Crypto Fear & Greed")
        crypto = fetch_crypto_fear_greed()
        if "error" in crypto:
            st.error(f"Could not fetch Crypto data: {crypto['error']}")
        else:
            sub = [
                ("Yesterday",   crypto["yesterday"]),
                ("1 Week Ago",  crypto["prev_week"]),
                ("1 Month Ago", crypto["prev_month"]),
            ]
            st.markdown(
                gauge_html(crypto["score"], f"{crypto['score']} — {crypto['rating']}", "Crypto Fear & Greed Index", sub),
                unsafe_allow_html=True
            )

            # 30-day sparkline
            st.markdown("**30-Day History**")
            if crypto.get("history"):
                hist_scores = [h[0] for h in crypto["history"]]
                hist_df = pd.DataFrame({"Fear & Greed": hist_scores})
                st.line_chart(hist_df, height=150, use_container_width=True)

    if st.button("Refresh Sentiment", key="sentiment_refresh"):
        fetch_cnn_fear_greed.clear()
        fetch_crypto_fear_greed.clear()
        fetch_sentiment_prices.clear()
        fetch_ibit_approx.clear()
        st.rerun()

    st.caption("Prices via Yahoo Finance. CNN/Crypto indices refresh every 30 min. IBIT ~ = estimated from live BTC price.")

    if sent_live_on:
        sent_countdown = st.empty()
        for i in range(st.session_state['sent_interval'], 0, -1):
            sent_countdown.markdown(
                f'<span style="color:#6B7280;font-size:0.75em;">Next refresh in {i}s</span>',
                unsafe_allow_html=True
            )
            time.sleep(1)
        sent_countdown.empty()
        fetch_sentiment_prices.clear()
        fetch_ibit_approx.clear()

def _get_rotation_state(strat_trades):
    """
    Inspect trades for a rotation strategy and return:
      active_leg    – int leg number currently open (has Rotate Into, no Rotate Out), or None
      active_ticker – ticker for the active leg, or None
      next_leg      – int leg number to use for the next Rotate Into
      all_closed    – True when every leg that was opened has also been closed
    """
    if strat_trades.empty:
        return None, None, 1, True
    rot_in  = strat_trades[strat_trades['event_type'] == 'Rotate Into']
    rot_out = strat_trades[strat_trades['event_type'] == 'Rotate Out']
    if rot_in.empty:
        return None, None, 1, True
    legs_in  = set(rot_in['leg'].dropna().astype(int).tolist())
    legs_out = set(rot_out['leg'].dropna().astype(int).tolist())
    open_legs = legs_in - legs_out
    max_leg = max(legs_in)
    if not open_legs:
        return None, None, max_leg + 1, True
    active_leg = max(open_legs)
    ticker_row = rot_in[rot_in['leg'] == active_leg]
    active_ticker = str(ticker_row.iloc[0]['ticker']).upper() if not ticker_row.empty else None
    return active_leg, active_ticker, max_leg + 1, False


# =====================================================================
# --- TRADES TAB ---
# =====================================================================
@st.fragment
def _trades_tab():
    st.markdown('<div style="font-size:1em;font-weight:700;color:#e2e8f0;padding:2px 0 10px;">Trade Journal</div>', unsafe_allow_html=True)

    trades = load_trades()
    tr_journal, tr_add, tr_perf = st.tabs(["Journal", "Add Trade", "Performance"])

    # ------------------------------------------------------------------
    # JOURNAL
    # ------------------------------------------------------------------
    with tr_journal:
        if trades.empty:
            st.info("No trades yet. Use **Add Trade** to log your first trade.")
        else:
            def _trade_card_html(row):
                et       = str(row.get('event_type', ''))
                color    = EVENT_COLORS.get(et, "#6b7280")
                dt       = row['date']
                date_str = dt.strftime('%b %d, %Y') if pd.notna(dt) else '—'
                is_opt   = et in _OPTION_EVENTS
                qty      = float(row.get('qty',  0) or 0)
                price    = float(row.get('price',0) or 0)
                strike_raw = row.get('strike')
                strike   = float(strike_raw) if pd.notna(strike_raw) and str(strike_raw) not in ('', 'nan') else None
                expiry   = str(row.get('expiry','')) if pd.notna(row.get('expiry')) else ''
                opt_t    = str(row.get('option_type','')) if pd.notna(row.get('option_type')) else ''
                notes    = str(row.get('notes','')) if pd.notna(row.get('notes')) and str(row.get('notes','')) not in ('', 'nan') else ''
                acct     = str(row.get('account_type','')) if pd.notna(row.get('account_type')) and str(row.get('account_type','')) not in ('', 'nan') else ''
                leg_val  = row.get('leg')

                if et == "Dividend":
                    detail = f"${price:,.2f} received (cash)"
                elif et == "Dividend (DRIP)":
                    detail = f"{qty:g} shares @ ${price:.2f} (reinvested)"
                elif is_opt:
                    parts = [f"{int(qty)} contract{'s' if qty!=1 else ''}"]
                    if strike: parts.append(f"${strike:g}{opt_t}")
                    if expiry: parts.append(f"exp {expiry}")
                    if price:  parts.append(f"@ ${price:.2f}/sh")
                    detail = " · ".join(parts)
                else:
                    detail = f"{int(qty)} shares @ ${price:.2f}"

                cf       = trade_cash_flow(row)
                cf_str   = f"+${cf:,.2f}" if cf >= 0 else f"-${abs(cf):,.2f}"
                cf_color = "#22c55e" if cf >= 0 else "#f87171"
                acct_badge = (
                    f'<span style="background:#1e3a2f;color:#4ade80;font-size:0.65em;'
                    f'padding:1px 6px;border-radius:6px;margin-left:6px;">{acct}</span>'
                ) if acct else ''
                leg_badge = (
                    f'<span style="background:#0c2233;color:#06b6d4;font-size:0.65em;'
                    f'padding:1px 6px;border-radius:6px;margin-left:6px;">Leg {int(leg_val)}</span>'
                ) if pd.notna(leg_val) and str(leg_val) not in ('', 'nan') else ''
                notes_html = (
                    f'<div style="color:#6b7280;font-size:0.75em;margin-top:3px;">{notes}</div>'
                ) if notes else ''

                return f"""
<div style="background:#151a27;border-left:3px solid {color};border-radius:6px;
            padding:8px 12px;margin-bottom:5px;">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px;">
    <div>
      <span style="color:{color};font-weight:700;font-size:0.85em;">{et}</span>{acct_badge}{leg_badge}
    </div>
    <div style="text-align:right;">
      <span style="color:{cf_color};font-weight:700;">{cf_str}</span>
      <span style="color:#6b7280;font-size:0.75em;margin-left:8px;">{date_str}</span>
    </div>
  </div>
  <div style="color:#94a3b8;font-size:0.8em;margin-top:2px;">{detail}</div>{notes_html}
</div>"""

            def _render_trade_row(row, trades_df):
                trade_id = int(row['id']) if pd.notna(row.get('id')) else abs(hash(str(row.get('date','')) + str(row.get('ticker',''))))
                edit_key = f'_tedit_{trade_id}'

                col_card, col_btn = st.columns([20, 1])
                with col_card:
                    st.markdown(_trade_card_html(row), unsafe_allow_html=True)
                with col_btn:
                    if st.button("✏", key=f"ebtn_{trade_id}", help="Edit trade"):
                        st.session_state[edit_key] = not st.session_state.get(edit_key, False)

                if st.session_state.get(edit_key, False):
                    with st.container():
                        st.markdown(
                            '<div style="background:#0f1621;border:1px solid #1e2a3f;border-radius:8px;'
                            'padding:12px 16px;margin-bottom:8px;">',
                            unsafe_allow_html=True
                        )
                        st.markdown('<div style="font-size:0.8em;font-weight:700;color:#94a3b8;margin-bottom:8px;">Edit Trade</div>', unsafe_allow_html=True)
                        with st.form(f"edit_form_{trade_id}", clear_on_submit=False):
                            r1a, r1b, r1c = st.columns(3)
                            _cur_date = row['date'].date() if pd.notna(row.get('date')) else datetime.today().date()
                            e_date     = r1a.date_input("Date", value=_cur_date, key=f"ed_dt_{trade_id}")
                            e_ticker   = r1b.text_input("Ticker", value=str(row.get('ticker', '')).upper(), key=f"ed_tk_{trade_id}").strip().upper()
                            _et_list   = TRADE_EVENTS
                            _cur_et    = str(row.get('event_type', ''))
                            _et_idx    = _et_list.index(_cur_et) if _cur_et in _et_list else 0
                            e_event    = r1c.selectbox("Event Type", _et_list, index=_et_idx, key=f"ed_ev_{trade_id}")

                            r2a, r2b, r2c = st.columns(3)
                            e_qty   = r2a.number_input("Qty / Shares", value=float(row.get('qty', 0) or 0), min_value=0.0, step=1.0, key=f"ed_qt_{trade_id}")
                            e_price = r2b.number_input("Price", value=float(row.get('price', 0) or 0), min_value=0.0, step=0.01, key=f"ed_pr_{trade_id}")
                            e_fees  = r2c.number_input("Fees", value=float(row.get('fees', 0) or 0), min_value=0.0, step=0.01, key=f"ed_fe_{trade_id}")

                            r3a, r3b, r3c = st.columns(3)
                            _stk_raw = row.get('strike')
                            if pd.notna(_stk_raw) and str(_stk_raw) not in ('', 'nan'):
                                _stk_f   = float(_stk_raw)
                                _stk_str = str(int(_stk_f)) if _stk_f == int(_stk_f) else str(_stk_f)
                            else:
                                _stk_str = ''
                            e_strike   = r3a.text_input("Strike", value=_stk_str, key=f"ed_sk_{trade_id}")
                            e_expiry   = r3b.text_input("Expiry (YYYY-MM-DD)", value=str(row.get('expiry', '')) if pd.notna(row.get('expiry')) and str(row.get('expiry', '')) != 'nan' else '', key=f"ed_ex_{trade_id}")
                            e_opt_type = r3c.text_input("Opt Type (C/P)", value=str(row.get('option_type', '')) if pd.notna(row.get('option_type')) and str(row.get('option_type', '')) not in ('', 'nan') else '', key=f"ed_op_{trade_id}").strip().upper()

                            r4a, r4b = st.columns(2)
                            _leg_raw = row.get('leg')
                            e_leg   = r4a.text_input("Leg", value=str(int(float(_leg_raw))) if pd.notna(_leg_raw) and str(_leg_raw) not in ('', 'nan') else '', key=f"ed_lg_{trade_id}")
                            e_notes = r4b.text_input("Notes", value=str(row.get('notes', '')) if pd.notna(row.get('notes')) and str(row.get('notes', '')) not in ('', 'nan') else '', key=f"ed_no_{trade_id}")

                            save_clicked = st.form_submit_button("Save Changes", type="primary")

                        st.markdown('</div>', unsafe_allow_html=True)

                        if save_clicked:
                            idx_list = trades_df.index[trades_df['id'] == trade_id].tolist()
                            if idx_list:
                                i = idx_list[0]
                                trades_df.at[i, 'date']        = str(e_date)
                                trades_df.at[i, 'ticker']      = e_ticker
                                trades_df.at[i, 'event_type']  = e_event
                                trades_df.at[i, 'qty']         = float(e_qty)
                                trades_df.at[i, 'price']       = float(e_price)
                                trades_df.at[i, 'fees']        = float(e_fees)
                                trades_df.at[i, 'strike']      = float(e_strike) if e_strike.strip() else np.nan
                                trades_df.at[i, 'expiry']      = e_expiry.strip()
                                trades_df.at[i, 'option_type'] = e_opt_type
                                trades_df.at[i, 'leg']         = int(float(e_leg)) if e_leg.strip() else np.nan
                                trades_df.at[i, 'notes']       = e_notes.strip()
                                save_trades(trades_df)
                                st.session_state[edit_key] = False
                                st.success("Trade updated.")
                                st.rerun()

            # Group by strategy
            strategies = sorted(trades['strategy'].dropna().astype(str).unique().tolist())
            for strat in strategies:
                strat_trades = trades[trades['strategy'] == strat].copy()
                strat_trades['_cf'] = strat_trades.apply(trade_cash_flow, axis=1)
                strat_cf     = strat_trades['_cf'].sum()
                n_trades     = len(strat_trades)
                cf_color     = "#22c55e" if strat_cf >= 0 else "#f87171"
                cf_str       = f"+${strat_cf:,.2f}" if strat_cf >= 0 else f"-${abs(strat_cf):,.2f}"
                is_rotation  = strat_trades['event_type'].isin(_ROTATION_EVENTS).any()

                if is_rotation:
                    rot_in_sorted = strat_trades[strat_trades['event_type'] == 'Rotate Into'].sort_values('leg', na_position='last')
                    ticker_chain  = " → ".join(str(r['ticker']).upper() for _, r in rot_in_sorted.iterrows()) or '?'
                    header_ticker = ticker_chain
                else:
                    header_ticker = str(strat_trades['ticker'].iloc[0]).upper() if not strat_trades.empty else ''

                with st.expander(
                    f"{strat}  ·  {header_ticker}  ·  {n_trades} trade{'s' if n_trades != 1 else ''}  ·  {cf_str}",
                    expanded=False
                ):
                    if is_rotation:
                        legs = sorted([int(l) for l in strat_trades['leg'].dropna().unique()])
                        rot_in_all = strat_trades[strat_trades['event_type'] == 'Rotate Into']
                        rot_out_all = strat_trades[strat_trades['event_type'] == 'Rotate Out']
                        for leg_num in legs:
                            leg_trades  = strat_trades[strat_trades['leg'] == leg_num].sort_values('date', ascending=False)
                            leg_cf      = leg_trades['_cf'].sum()
                            leg_cf_str  = f"+${leg_cf:,.2f}" if leg_cf >= 0 else f"-${abs(leg_cf):,.2f}"
                            leg_cf_col  = "#22c55e" if leg_cf >= 0 else "#f87171"
                            leg_in_row  = rot_in_all[rot_in_all['leg'] == leg_num]
                            leg_out_row = rot_out_all[rot_out_all['leg'] == leg_num]
                            leg_ticker  = str(leg_in_row.iloc[0]['ticker']).upper() if not leg_in_row.empty else '?'
                            status      = "closed" if not leg_out_row.empty else "open"
                            status_col  = "#6b7280" if status == "closed" else "#22c55e"
                            st.markdown(
                                f'<div style="font-size:0.78em;font-weight:700;color:#94a3b8;'
                                f'padding:6px 0 3px;border-bottom:1px solid #1e2a3f;margin-bottom:6px;">'
                                f'Leg {leg_num} — {leg_ticker} &nbsp;'
                                f'<span style="color:{leg_cf_col};">{leg_cf_str}</span> &nbsp;'
                                f'<span style="color:{status_col};font-weight:400;">({status})</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            for _, row in leg_trades.iterrows():
                                _render_trade_row(row, trades)
                        # Trades without a leg (shouldn't happen, but handle gracefully)
                        unassigned = strat_trades[strat_trades['leg'].isna()].sort_values('date', ascending=False)
                        if not unassigned.empty:
                            st.markdown('<div style="font-size:0.75em;color:#6b7280;padding:4px 0;">Unassigned</div>', unsafe_allow_html=True)
                            for _, row in unassigned.iterrows():
                                _render_trade_row(row, trades)
                    else:
                        for _, row in strat_trades.sort_values('date', ascending=False).iterrows():
                            _render_trade_row(row, trades)

            st.divider()
            with st.expander("Delete a Trade"):
                def _tr_del_label(r):
                    dt_s = r['date'].strftime('%Y-%m-%d') if pd.notna(r['date']) else '?'
                    return f"{dt_s} | {r['event_type']} | {str(r['ticker']).upper()} | id#{int(r['id']) if pd.notna(r.get('id')) else '?'}"
                del_labels = [_tr_del_label(r) for _, r in trades.iterrows()]
                del_sel = st.selectbox("Select trade to delete", del_labels, key="tr_del_sel")
                del_idx = del_labels.index(del_sel)
                st.warning(f"Permanently delete: **{del_sel}**?")
                if st.button("Delete Trade", type="primary", key="tr_del_btn"):
                    save_trades(trades.drop(index=del_idx).reset_index(drop=True))
                    st.success("Trade deleted.")
                    st.rerun()

    # ------------------------------------------------------------------
    # ADD TRADE
    # ------------------------------------------------------------------
    with tr_add:
        existing_strategies = (
            sorted(trades['strategy'].dropna().astype(str).unique().tolist())
            if not trades.empty else []
        )

        event_type = st.selectbox("Event Type", TRADE_EVENTS, key="tr_add_event")
        is_opt_sell  = event_type in _OPTION_SELL_EVENTS
        is_opt_close = event_type in _OPTION_CLOSE_EVENTS
        is_assign    = event_type in _ASSIGN_EVENTS
        is_expired   = event_type == "Expired Worthless"
        is_dividend  = event_type in {"Dividend", "Dividend (DRIP)"}
        is_drip      = event_type == "Dividend (DRIP)"
        is_rotation  = event_type in _ROTATION_EVENTS

        # For assignment / close / expiry: pick the originating sell trade
        # and pre-fill everything from it.
        _ref_source = {
            "Assigned (Put)":        "Sell Cash-Secured Put",
            "Assigned (Call)":       "Sell Covered Call",
            "Buy to Close (Put)":    "Sell Cash-Secured Put",
            "Buy to Close (Call)":   "Sell Covered Call",
            "Expired Worthless":     None,   # either type — show all sell trades
        }
        needs_ref = event_type in _ref_source

        # Build the list of open sell-option trades the user can reference
        if is_rotation:
            is_rotate_out = event_type == "Rotate Out"
            is_rotate_in  = event_type == "Rotate Into"

            # Rotation strategy picker (outside form so state can react)
            rot_strat_options = ["(new strategy)"] + existing_strategies
            rot_strat_sel = st.selectbox("Strategy", rot_strat_options, key="tr_rot_strat_sel")

            rot_active_leg, rot_active_ticker, rot_next_leg = None, None, 1
            rot_can_proceed = True

            if rot_strat_sel == "(new strategy)":
                if is_rotate_out:
                    st.error("Cannot Rotate Out on a new strategy — start with **Rotate Into**.")
                    rot_can_proceed = False
                else:
                    rot_next_leg = 1
            else:
                if not trades.empty:
                    s_trades = trades[trades['strategy'] == rot_strat_sel].copy()
                    rot_active_leg, rot_active_ticker, rot_next_leg, _ = _get_rotation_state(s_trades)

                if is_rotate_in:
                    if rot_active_leg is not None:
                        st.error(
                            f"Leg {rot_active_leg} ({rot_active_ticker}) is still open. "
                            f"Log a **Rotate Out** first before starting leg {rot_next_leg}."
                        )
                        rot_can_proceed = False
                    elif rot_next_leg > 1:
                        st.info(f"Previous leg closed — starting leg {rot_next_leg}.")

                elif is_rotate_out:
                    if rot_active_leg is None:
                        st.error("No open rotation leg found for this strategy. Log a **Rotate Into** first.")
                        rot_can_proceed = False
                    else:
                        st.info(f"Rotating out of leg {rot_active_leg} — **{rot_active_ticker}**")

            if rot_can_proceed:
                if is_rotate_in:
                    if rot_strat_sel == "(new strategy)":
                        c1, c2 = st.columns([3, 2])
                        tr_rot_strategy = c1.text_input("New strategy name", placeholder="e.g. Rotation-AAPL-META", key="tr_rot_strat_new").strip()
                        tr_rot_ticker   = c2.text_input("Ticker (rotating into)", placeholder="META", key="tr_rot_ticker_new").strip().upper()
                    else:
                        tr_rot_strategy = rot_strat_sel
                        tr_rot_ticker   = st.text_input("Ticker (rotating into)", placeholder="META", key="tr_rot_ticker_in").strip().upper()
                    tr_rot_leg = rot_next_leg
                    st.caption(f"This will be logged as **Leg {tr_rot_leg}**.")
                else:  # Rotate Out
                    tr_rot_strategy = rot_strat_sel
                    tr_rot_ticker   = rot_active_ticker or ""
                    tr_rot_leg      = rot_active_leg

                with st.form("tr_add_form_rot", clear_on_submit=True):
                    rc1, rc2 = st.columns(2)
                    tr_rot_date = rc1.date_input("Date", value=datetime.today(), key="tr_rot_date_f")
                    tr_rot_qty  = rc2.number_input("Shares", min_value=1, step=1, value=100, key="tr_rot_qty_f")
                    rp1, rp2   = st.columns(2)
                    tr_rot_price = rp1.number_input("Price per share ($)", min_value=0.01, step=0.01, value=100.0, key="tr_rot_price_f")
                    tr_rot_fees  = rp2.number_input("Fees ($)", min_value=0.0, step=0.01, value=0.0, key="tr_rot_fees_f")
                    tr_rot_notes = st.text_input("Notes (optional)", key="tr_rot_notes_f").strip()
                    rot_submitted = st.form_submit_button("Log Trade", type="primary")

                if rot_submitted:
                    if not tr_rot_strategy or tr_rot_strategy == "(new strategy)":
                        st.error("Strategy name is required.")
                    elif not tr_rot_ticker:
                        st.error("Ticker is required.")
                    else:
                        next_id = int(trades['id'].max() + 1) if not trades.empty and pd.notna(trades['id']).any() else 1
                        new_row = pd.DataFrame([{
                            'id':               next_id,
                            'date':             str(tr_rot_date),
                            'ticker':           tr_rot_ticker,
                            'strategy':         tr_rot_strategy,
                            'event_type':       event_type,
                            'qty':              float(tr_rot_qty),
                            'strike':           np.nan,
                            'expiry':           '',
                            'option_type':      '',
                            'price':            float(tr_rot_price),
                            'fees':             float(tr_rot_fees),
                            'account_type':     '',
                            'capital_reserved': np.nan,
                            'notes':            tr_rot_notes,
                            'leg':              int(tr_rot_leg),
                        }])
                        save_trades(pd.concat([trades, new_row], ignore_index=True))
                        cf = trade_cash_flow(new_row.iloc[0])
                        cf_str = f"+${cf:,.2f}" if cf >= 0 else f"-${abs(cf):,.2f}"
                        st.success(f"Logged: {event_type} — {tr_rot_ticker} Leg {tr_rot_leg} ({cf_str})")
                        st.rerun()

        elif needs_ref:
            src_event = _ref_source[event_type]
            if src_event:
                ref_pool = trades[trades['event_type'] == src_event].copy()
            else:
                ref_pool = trades[trades['event_type'].isin(_OPTION_SELL_EVENTS)].copy()

            def _ref_label(r):
                exp = str(r.get('expiry','')) if pd.notna(r.get('expiry')) else '?'
                stk = f"${float(r['strike']):g}" if pd.notna(r.get('strike')) else '?'
                ot  = str(r.get('option_type',''))
                qty = int(r.get('qty', 0))
                return (f"{str(r['ticker']).upper()} {ot} {stk} exp {exp} "
                        f"({qty} contract{'s' if qty!=1 else ''}) — {r['strategy']}")

            if ref_pool.empty:
                st.warning(f"No open '{src_event or 'sell option'}' trades found to link. "
                           "Log the original sell trade first.")
                ref_row = None
            else:
                ref_labels = [_ref_label(r) for _, r in ref_pool.iterrows()]
                ref_sel    = st.selectbox("Which trade?", ref_labels, key="tr_add_ref_sel")
                ref_idx    = ref_labels.index(ref_sel)
                ref_row    = ref_pool.iloc[ref_idx]

            if ref_row is not None:
                # For assignment/expiry: date = expiry from the original trade, fees already paid
                _expiry_str = str(ref_row.get('expiry', ''))
                _auto_date  = None
                if _expiry_str:
                    try:
                        _auto_date = datetime.strptime(_expiry_str, '%Y-%m-%d').date()
                    except ValueError:
                        pass

                if is_opt_close:
                    # Buy to Close: still need date + close price + optional fees
                    with st.form("tr_add_form", clear_on_submit=True):
                        fe1, fe2, fe3 = st.columns(3)
                        tr_date  = fe1.date_input("Date", value=datetime.today(), key="tr_add_date_f")
                        tr_price = fe2.number_input("Close price ($/share as quoted)", min_value=0.0, step=0.01, value=0.0, key="tr_add_close_price_f")
                        tr_fees  = fe3.number_input("Fees ($)", min_value=0.0, step=0.01, value=0.0, key="tr_add_fees_f")
                        tr_notes = st.text_input("Notes (optional)", key="tr_add_notes_f").strip()
                        submitted = st.form_submit_button("Log Trade", type="primary")
                else:
                    # Assigned or Expired: date auto-set to expiry, no fees, no price
                    _date_label = f"Expiry date from original trade: **{_expiry_str}**" if _expiry_str else "Expiry date unknown"
                    st.caption(_date_label)
                    tr_date  = _auto_date or datetime.today().date()
                    tr_price = 0.0
                    tr_fees  = 0.0
                    with st.form("tr_add_form", clear_on_submit=True):
                        tr_notes  = st.text_input("Notes (optional)", key="tr_add_notes_f").strip()
                        submitted = st.form_submit_button("Log Trade", type="primary")

                if submitted:
                    next_id = int(trades['id'].max() + 1) if not trades.empty and pd.notna(trades['id']).any() else 1
                    _ref_leg = ref_row.get('leg') if pd.notna(ref_row.get('leg')) else np.nan
                    new_row = pd.DataFrame([{
                        'id':          next_id,
                        'date':        str(tr_date),
                        'ticker':      ref_row['ticker'],
                        'strategy':    ref_row['strategy'],
                        'event_type':  event_type,
                        'qty':         float(ref_row['qty']),
                        'strike':      float(ref_row['strike']) if pd.notna(ref_row.get('strike')) else np.nan,
                        'expiry':      _expiry_str,
                        'option_type': str(ref_row.get('option_type', '')),
                        'price':       float(tr_price),
                        'fees':        float(tr_fees),
                        'notes':       tr_notes,
                        'leg':         _ref_leg,
                    }])
                    save_trades(pd.concat([trades, new_row], ignore_index=True))
                    cf = trade_cash_flow(new_row.iloc[0])
                    cf_str = f"+${cf:,.2f}" if cf >= 0 else f"-${abs(cf):,.2f}"
                    st.success(f"Logged: {event_type} — {str(ref_row['ticker']).upper()} ({cf_str})")
                    st.rerun()

        else:
            # Standard form: Buy/Sell Stock, Sell Covered Call, Sell CSP, Dividend
            # Build strategy→ticker lookup from existing trades
            strat_ticker_map = {}
            if not trades.empty:
                for _, _r in trades.iterrows():
                    _s = str(_r.get('strategy', '')).strip()
                    _t = str(_r.get('ticker', '')).strip().upper()
                    if _s and _t and _s not in strat_ticker_map:
                        strat_ticker_map[_s] = _t

            # Strategy picker OUTSIDE the form so ticker can react immediately
            strat_options = ["(new strategy)"] + existing_strategies
            strat_sel = st.selectbox("Strategy", strat_options, key="tr_add_strat_sel_f")

            if strat_sel == "(new strategy)":
                c_strat, c_ticker = st.columns([3, 2])
                tr_strategy = c_strat.text_input("New strategy name", placeholder="e.g. AAPL Covered Call Wheel", key="tr_add_strat_new_f").strip()
                tr_ticker   = c_ticker.text_input("Ticker", placeholder="AAPL", key="tr_add_ticker_f").strip().upper()
            else:
                tr_strategy    = strat_sel
                implied_ticker = strat_ticker_map.get(strat_sel, "")
                st.caption(f"Ticker: **{implied_ticker}**")
                tr_ticker = implied_ticker

            # For rotation strategies: show leg selector and override ticker to leg-specific one
            tr_leg = np.nan
            if tr_strategy and tr_strategy != "(new strategy)" and not trades.empty:
                s_trades = trades[trades['strategy'] == tr_strategy]
                if s_trades['event_type'].isin(_ROTATION_EVENTS).any():
                    rot_in_trades = s_trades[s_trades['event_type'] == 'Rotate Into']
                    available_legs = sorted([int(l) for l in rot_in_trades['leg'].dropna().unique()])
                    if available_legs:
                        leg_labels = []
                        for lg in available_legs:
                            lg_ticker = str(rot_in_trades[rot_in_trades['leg'] == lg].iloc[0]['ticker']).upper()
                            leg_labels.append(f"Leg {lg} — {lg_ticker}")
                        _al, _, _, _ = _get_rotation_state(s_trades)
                        default_idx = available_legs.index(_al) if _al in available_legs else len(available_legs) - 1
                        leg_sel = st.selectbox("Rotation Leg", leg_labels, index=default_idx, key="tr_add_leg_sel")
                        tr_leg = available_legs[leg_labels.index(leg_sel)]
                        # Override ticker to match the selected leg
                        leg_ticker_row = rot_in_trades[rot_in_trades['leg'] == tr_leg]
                        if not leg_ticker_row.empty:
                            tr_ticker = str(leg_ticker_row.iloc[0]['ticker']).upper()
                            st.caption(f"Ticker for this leg: **{tr_ticker}**")

            if event_type == "Sell Covered Call" and existing_strategies:
                st.info(
                    "Pick the **same strategy** as your stock/assignment trade so P&L rolls up correctly.",
                    icon="💡"
                )

            with st.form("tr_add_form", clear_on_submit=True):
                tr_date = st.date_input("Date", value=datetime.today(), key="tr_add_date_f")

                if is_dividend:
                    # Auto-compute shares held for this strategy/ticker from open positions
                    _div_open = compute_open_stock_positions(
                        trades[trades['strategy'] == tr_strategy] if tr_strategy and tr_strategy != "(new strategy)" else trades
                    )
                    _div_shares = _div_open.get(tr_ticker, {}).get('shares', 0) if tr_ticker else 0

                    if is_drip:
                        # DRIP: only care about shares added and price for cost basis
                        st.caption(f"Shares currently held: **{int(_div_shares)}**" if _div_shares > 0 else "No open shares found for this strategy/ticker.")
                        dp1, dp2 = st.columns(2)
                        tr_qty   = dp1.number_input("New shares received (DRIP)", min_value=0.0, step=0.0001, value=0.0, format="%.6f", key="tr_add_drip_qty_f")
                        tr_price = dp2.number_input("Price per share ($)", min_value=0.0, step=0.01, value=0.0, key="tr_add_drip_price_f")
                    else:
                        # Cash dividend
                        d1, d2 = st.columns(2)
                        tr_div_per_share = d1.number_input(
                            "Dividend per share ($)", min_value=0.0, step=0.001, value=0.0,
                            format="%.3f", key="tr_add_div_ps_f"
                        )
                        if _div_shares > 0:
                            _div_total = tr_div_per_share * _div_shares
                            d2.metric("Total dividend", f"${_div_total:,.2f}",
                                      delta=f"{int(_div_shares)} shares")
                            tr_price = _div_total
                        else:
                            tr_price = d2.number_input(
                                "Or enter total amount ($)", min_value=0.0, step=0.01,
                                value=0.0, key="tr_add_div_total_f"
                            )
                            if tr_div_per_share > 0:
                                tr_price = tr_div_per_share  # fallback
                        tr_qty = float(_div_shares) if _div_shares > 0 else 1.0

                    tr_strike   = np.nan
                    tr_expiry   = ""
                    tr_opt_type = ""
                    tr_acct     = ""
                    tr_cap_res  = np.nan

                elif is_opt_sell:
                    # Option type is implied by event name — no selector needed
                    tr_opt_type = "C" if event_type == "Sell Covered Call" else "P"
                    nc1, nc2    = st.columns(2)
                    tr_qty      = nc1.number_input("Contracts", min_value=1, step=1, value=1, key="tr_add_qty_f")
                    tr_strike   = nc2.number_input("Strike ($)", min_value=0.01, step=0.5, value=100.0, key="tr_add_strike_f")
                    pd1, pd2    = st.columns(2)
                    tr_expiry   = pd1.text_input("Expiry (YYYY-MM-DD)", placeholder="2025-05-16", key="tr_add_expiry_f").strip()
                    tr_price    = pd2.number_input("Premium ($/share as quoted)", min_value=0.0, step=0.01, value=0.0, key="tr_add_price_opt_f")

                    if event_type == "Sell Cash-Secured Put":
                        tr_acct    = st.radio("Account type", ["Cash", "Margin"], horizontal=True, key="tr_add_acct_f")
                        _notional  = float(tr_qty) * tr_strike * 100
                        tr_cap_res = _notional   # always use notional for benchmark comparison
                        if tr_acct == "Cash":
                            st.caption(f"Capital reserved: **${_notional:,.2f}** (strike × 100 × contracts)")
                        else:
                            st.caption(
                                f"Benchmark will use notional **${_notional:,.2f}** (strike × 100 × contracts) "
                                "as your economic exposure — no cash reservation needed on margin."
                            )
                    else:
                        tr_acct    = ""
                        tr_cap_res = np.nan

                else:  # Buy Stock / Sell Stock
                    sc1, sc2    = st.columns(2)
                    tr_qty      = sc1.number_input("Shares", min_value=1, step=1, value=100, key="tr_add_qty_stk_f")
                    tr_price    = sc2.number_input("Price per share ($)", min_value=0.01, step=0.01, value=100.0, key="tr_add_price_stk_f")
                    tr_strike   = np.nan
                    tr_expiry   = ""
                    tr_opt_type = ""
                    tr_acct     = ""
                    tr_cap_res  = np.nan

                fe1, fe2 = st.columns([2, 3])
                tr_fees  = fe1.number_input("Fees / commission ($)", min_value=0.0, step=0.01, value=0.0, key="tr_add_fees_f")
                tr_notes = fe2.text_input("Notes (optional)", key="tr_add_notes_f").strip()

                submitted = st.form_submit_button("Log Trade", type="primary")

            if submitted:
                if not tr_ticker:
                    st.error("Ticker is required — enter it alongside your new strategy name.")
                elif not tr_strategy:
                    st.error("Strategy name is required.")
                else:
                    next_id = int(trades['id'].max() + 1) if not trades.empty and pd.notna(trades['id']).any() else 1
                    new_row = pd.DataFrame([{
                        'id':               next_id,
                        'date':             str(tr_date),
                        'ticker':           tr_ticker,
                        'strategy':         tr_strategy,
                        'event_type':       event_type,
                        'qty':              float(tr_qty),
                        'strike':           float(tr_strike) if pd.notna(tr_strike) else np.nan,
                        'expiry':           tr_expiry,
                        'option_type':      tr_opt_type,
                        'price':            float(tr_price),
                        'fees':             float(tr_fees),
                        'account_type':     tr_acct,
                        'capital_reserved': float(tr_cap_res) if pd.notna(tr_cap_res) else np.nan,
                        'notes':            tr_notes,
                        'leg':              int(tr_leg) if pd.notna(tr_leg) else np.nan,
                    }])
                    save_trades(pd.concat([trades, new_row], ignore_index=True))
                    cf = trade_cash_flow(new_row.iloc[0])
                    cf_str = f"+${cf:,.2f}" if cf >= 0 else f"-${abs(cf):,.2f}"
                    st.success(f"Logged: {event_type} — {tr_ticker} ({cf_str})")
                    st.rerun()

    # ------------------------------------------------------------------
    # PERFORMANCE
    # ------------------------------------------------------------------
    with tr_perf:
        if trades.empty:
            st.info("No trades yet. Add some trades to see performance.")
        else:
            all_strategies = sorted(trades['strategy'].dropna().astype(str).unique().tolist())
            perf_strat = st.selectbox(
                "Strategy", ["All"] + all_strategies, key="perf_strat_sel"
            )
            perf_trades = trades if perf_strat == "All" else trades[trades['strategy'] == perf_strat]

            trades_cf = perf_trades.copy()
            trades_cf['cash_flow'] = trades_cf.apply(trade_cash_flow, axis=1)

            # --- Open stock positions + live prices ---
            open_positions = compute_open_stock_positions(perf_trades)
            live_prices = {}
            if open_positions:
                live_prices = fetch_trade_live_prices(tuple(sorted(open_positions.keys())))

            total_mkt_value  = 0.0
            total_cost_basis = 0.0
            for tkr, pos in open_positions.items():
                px = live_prices.get(tkr)
                if px:
                    total_mkt_value  += px * pos['shares']
                    total_cost_basis += pos['cost_basis']
            total_unrealized = total_mkt_value - total_cost_basis
            total_premium    = trades_cf[trades_cf['event_type'].isin(_OPTION_SELL_EVENTS)]['cash_flow'].sum()
            total_realized   = trades_cf['cash_flow'].sum()
            # Total P&L = net cash flows + current value of shares still held
            total_pnl        = total_realized + total_mkt_value

            # --- Top-line metrics ---
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Premium Collected", f"${total_premium:,.2f}")
            m2.metric("Unrealized (Open Shares)", f"${total_unrealized:+,.2f}",
                      delta=f"${total_mkt_value:,.2f} mkt value" if total_mkt_value else "no open positions")
            m3.metric("Total P&L (vs Capital Deployed)", f"${total_pnl:,.2f}",
                      delta=f"${total_cost_basis:,.2f} deployed" if total_cost_basis else None)

            # --- Open positions table ---
            if open_positions:
                st.divider()
                st.markdown("**Open Stock Positions**")
                pos_rows = []
                for tkr, pos in open_positions.items():
                    px  = live_prices.get(tkr)
                    avg = pos['cost_basis'] / pos['shares'] if pos['shares'] else 0
                    mkt = px * pos['shares'] if px else None
                    unr = (mkt - pos['cost_basis']) if mkt is not None else None
                    pos_rows.append({
                        "Ticker":        tkr,
                        "Shares":        f"{pos['shares']:g}",
                        "Avg Cost":      f"${avg:.2f}",
                        "Current Price": f"${px:.2f}" if px else "—",
                        "Mkt Value":     f"${mkt:,.2f}" if mkt is not None else "—",
                        "Unrealized":    f"${unr:+,.2f}" if unr is not None else "—",
                    })
                st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

            # --- Benchmark comparison ---
            # Rules for capital deployment:
            #   Cash CSP:     capital = capital_reserved at SELL date (cash locked up then)
            #   Margin CSP:   capital = strike×100×qty at ASSIGNMENT date (no cash until assigned)
            #   Buy Stock / Rotate Into: capital = qty×price at BUY date
            # Assignments from cash CSPs are SKIPPED — same cash already counted at sell date.
            # Each tranche is benchmarked from its own deployment date (not all from earliest).
            st.divider()
            st.markdown("**vs Benchmarks — per-tranche capital deployment**")

            sorted_trades = trades_cf.sort_values('date').dropna(subset=['date'])

            sell_trades   = sorted_trades[sorted_trades['event_type'].isin(_OPTION_SELL_EVENTS)].copy()
            assign_trades = sorted_trades[sorted_trades['event_type'].isin(_ASSIGN_EVENTS)].copy()
            buy_trades    = sorted_trades[sorted_trades['event_type'].isin({"Buy Stock", "Rotate Into"})].copy()

            cash_sells    = sell_trades[sell_trades['account_type'].fillna('') == 'Cash'] if 'account_type' in sell_trades.columns else pd.DataFrame()
            margin_sells  = sell_trades[sell_trades['account_type'].fillna('') == 'Margin'] if 'account_type' in sell_trades.columns else pd.DataFrame()

            # Strategies where cash was reserved at sell time — don't double-count their assignments
            cash_sell_strats = set(cash_sells['strategy'].tolist()) if not cash_sells.empty else set()

            # Build list of (date, capital, label) deployment events
            deployments = []

            # Cash CSP: capital locked at sell date
            for _, r in (cash_sells.iterrows() if not cash_sells.empty else []):
                cap = float(r['capital_reserved']) if pd.notna(r.get('capital_reserved')) else 0.0
                if cap > 0:
                    deployments.append({'date': pd.to_datetime(r['date']), 'capital': cap, 'label': f"CSP sell {str(r.get('ticker','')).upper()}"})

            # Margin CSP: capital deployed at assignment date; skip if strategy already has a cash sell
            for _, r in (assign_trades.iterrows() if not assign_trades.empty else []):
                if str(r.get('strategy', '')) in cash_sell_strats:
                    continue  # cash CSP — capital already counted at sell date
                strike_r = float(r['strike']) if pd.notna(r.get('strike')) else 0.0
                qty_r    = float(r['qty'])     if pd.notna(r.get('qty'))    else 0.0
                cap      = strike_r * 100 * qty_r
                if cap > 0:
                    deployments.append({'date': pd.to_datetime(r['date']), 'capital': cap, 'label': f"Assigned {str(r.get('ticker','')).upper()}"})

            # Buy Stock / Rotate Into: capital deployed at buy date
            for _, r in (buy_trades.iterrows() if not buy_trades.empty else []):
                qty_r   = float(r['qty'])   if pd.notna(r.get('qty'))   else 0.0
                price_r = float(r['price']) if pd.notna(r.get('price')) else 0.0
                cap     = qty_r * price_r
                if cap > 0:
                    deployments.append({'date': pd.to_datetime(r['date']), 'capital': cap, 'label': f"Buy {str(r.get('ticker','')).upper()}"})

            # Fallback: if nothing tagged yet, use total cost basis
            if not deployments and total_cost_basis > 0:
                first_trade_date = sorted_trades['date'].iloc[0] if not sorted_trades.empty else None
                if first_trade_date is not None:
                    deployments.append({'date': pd.to_datetime(first_trade_date), 'capital': total_cost_basis, 'label': 'Cost basis'})

            if deployments:
                dep_df     = pd.DataFrame(deployments).sort_values('date')
                first_date = dep_df['date'].iloc[0]
                capital    = float(dep_df['capital'].sum())
            else:
                first_date = None
                capital    = 0.0

            if first_date and capital > 0:
                first_date_str = first_date.strftime('%Y-%m-%d')

                with st.spinner("Fetching benchmark data..."):
                    bench = fetch_benchmark_history(("VTI", "QQQ", "BRK-B", "VXUS", "FREL"), first_date_str)

                if bench:
                    # Per-tranche benchmark: each deployment earns ETF return from its own date
                    bench_gain_by_etf = {}
                    for etf, series in bench.items():
                        series = series.dropna()
                        tranche_gain = 0.0
                        for _, dep_row in dep_df.iterrows():
                            s_from = series[series.index >= dep_row['date']]
                            if len(s_from) < 2:
                                continue
                            s0 = float(s_from.iloc[0])
                            s1 = float(series.iloc[-1])
                            tranche_gain += dep_row['capital'] * (s1 - s0) / s0
                        bench_gain_by_etf[etf] = tranche_gain

                    # --- Deployment breakdown ---
                    st.caption(f"**{len(dep_df)} capital deployment tranche{'s' if len(dep_df)>1 else ''}** · total ${capital:,.2f}")
                    dep_display_rows = []
                    for _, dep_row in dep_df.iterrows():
                        row_d = {
                            'Date':    dep_row['date'].strftime('%Y-%m-%d'),
                            'Event':   dep_row['label'],
                            'Capital': f"${dep_row['capital']:,.2f}",
                        }
                        for etf_name, etf_series in bench.items():
                            etf_s = etf_series.dropna()
                            s_from = etf_s[etf_s.index >= dep_row['date']]
                            if not s_from.empty:
                                buy_price   = float(s_from.iloc[0])
                                shares      = dep_row['capital'] / buy_price
                                today_price = float(etf_s.iloc[-1])
                                today_val   = shares * today_price
                                row_d[f"{etf_name} buy px"] = f"${buy_price:.2f}"
                                row_d[f"{etf_name} shares"] = f"{shares:.2f}"
                                row_d[f"{etf_name} value now"] = f"${today_val:,.2f}"
                            else:
                                row_d[f"{etf_name} buy px"]    = "—"
                                row_d[f"{etf_name} shares"]    = "—"
                                row_d[f"{etf_name} value now"] = "—"
                        dep_display_rows.append(row_d)
                    st.dataframe(pd.DataFrame(dep_display_rows), use_container_width=True, hide_index=True)

                    # --- Comparison table ---
                    bench_rows = []
                    for tkr, series in bench.items():
                        if tkr not in bench_gain_by_etf:
                            continue
                        bench_gain = bench_gain_by_etf[tkr]
                        bench_val  = capital + bench_gain
                        ret_pct    = bench_gain / capital if capital else 0
                        bench_rows.append({
                            "ETF":              tkr,
                            "Capital Deployed": f"${capital:,.2f}",
                            "ETF Value Today":  f"${bench_val:,.2f}",
                            "ETF Gain":         f"${bench_gain:+,.2f}",
                            "ETF Return":       f"{ret_pct*100:+.1f}%",
                            "Your Total P&L":   f"${total_pnl:,.2f}",
                            "You vs ETF":       f"${total_pnl - bench_gain:+,.2f}",
                        })

                    if bench_rows:
                        bench_df = pd.DataFrame(bench_rows)
                        st.dataframe(bench_df, use_container_width=True, hide_index=True)

                    # --- Equity curve: your running P&L vs benchmark growth ---
                    st.markdown("**Equity curve — your strategy vs benchmarks**")

                    # Build daily running P&L series from trades
                    cf_daily = (
                        sorted_trades
                        .assign(date=lambda d: pd.to_datetime(d['date']).dt.normalize())
                        .groupby('date')['cash_flow']
                        .sum()
                        .sort_index()
                    )
                    # Running realized cash (negative = money out, positive = money in)
                    # We flip sign for "portfolio value" view: start at capital deployed, track gains
                    running_cf = cf_daily.cumsum()

                    # Benchmark: grow capital at ETF daily returns
                    plot_data = {"Your Strategy": running_cf}
                    for tkr, series in bench.items():
                        series = series.dropna()
                        if len(series) < 2:
                            continue
                        # Normalize to same capital basis: gain/loss in dollars
                        norm = ((series / series.iloc[0]) - 1) * capital
                        norm.index = pd.to_datetime(norm.index).normalize()
                        plot_data[tkr] = norm

                    plot_df = pd.DataFrame(plot_data).sort_index()
                    plot_df.index.name = "Date"
                    # Forward-fill benchmark gaps (weekends/holidays)
                    plot_df = plot_df.ffill()
                    st.line_chart(plot_df, use_container_width=True, height=250)

                    st.caption(
                        f"Capital deployed: **${capital:,.2f}** across {len(dep_df)} tranche{'s' if len(dep_df)>1 else ''} "
                        f"(earliest: {first_date_str}). "
                        f"Your total P&L = realized cash flows (${total_realized:,.2f}) "
                        f"+ market value of open shares (${total_mkt_value:,.2f}). "
                        "Live prices refresh every 2 min. Benchmark uses per-tranche returns."
                    )
                else:
                    st.warning("Could not fetch benchmark data. Check your internet connection.")
            elif first_date and capital == 0:
                st.info("Benchmark comparison shows once capital is deployed (sell a put or get assigned).")
            else:
                st.info("Add trades to see benchmark comparison.")

with page_tab5:
    _trades_tab()
