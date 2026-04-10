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

def _yf_fetch_with_retry(fn, retries=3, base_delay=5):
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

@st.cache_data(ttl=120)
def fetch_option_data(occ_list):
    """Fetches delayed prices and IV from Yahoo Finance, batched by ticker."""
    results = []
    r = 0.045

    # Group OCCs by ticker so we fetch each ticker's spot/chain only once
    from collections import defaultdict
    by_ticker = defaultdict(list)
    for occ in occ_list:
        parsed = parse_occ(occ)
        if parsed:
            by_ticker[parsed[0]].append((occ, parsed))

    for i, (ticker_sym, contracts) in enumerate(by_ticker.items()):
        # Small delay between tickers to avoid rate limiting
        if i > 0:
            time.sleep(0.5)
        try:
            underlying_ticker = yf.Ticker(ticker_sym)
            
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

            # Fetch each unique expiration once per ticker
            chains = {}
            for occ, (_, expiration, _, _) in contracts:
                if expiration not in chains:
                    try:
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
                        st.warning(f"Could not find contract for {occ}. Strike {strike} may not be available (spot: {spot_price:.2f}).")
                        continue

                    fetched_option_price = contract['lastPrice'].values[0]
                    last_price = fetched_option_price
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

def construct_occ_from_row(row):
    """Constructs a standard OCC option symbol from a DataFrame row."""
    ticker = row['Ticker']
    exp_str = str(row['ExpirationYYMMDD'])
    yy = exp_str[:2]
    mm = exp_str[2:4]
    dd = exp_str[4:]
    opt_type = row['OptionType']
    strike = row['Strike']
    strike_formatted = f"{int(strike * 1000):08d}"
    return f"{ticker.upper()}{yy}{mm}{dd}{opt_type.upper()}{strike_formatted}"

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
page_tab1, page_tab2, page_tab3, page_tab4 = st.tabs(["Portfolio", "Watchlist", "Sentiment", "Summary"])

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
            df = pd.read_csv(file, header=None, skiprows=1, names=column_names)
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

    pf_ctrl1, pf_ctrl2, pf_ctrl3 = st.columns([2, 2, 4])
    pf_live_on = pf_ctrl1.toggle("Live Quotes", value=st.session_state['pf_live'], key="pf_live_toggle")
    st.session_state['pf_live'] = pf_live_on
    if pf_live_on:
        pf_interval = pf_ctrl2.selectbox("Refresh every", [15, 30, 60, 120], index=1, format_func=lambda x: f"{x}s", key="pf_interval_sel")
        st.session_state['pf_interval'] = pf_interval

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

@st.cache_data(ttl=120)
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
            underlying_ticker = yf.Ticker(ticker)
            spot, stock_date = get_latest_price(underlying_ticker)
            chain = underlying_ticker.option_chain(expiration)
            options = chain.calls if opt_type == 'C' else chain.puts
            contract = options[options['strike'] == strike]
            if contract.empty:
                results[occ] = {'option_price': None, 'spot': spot, 'delta': None, 'gamma': None}
                continue
            option_price = float(contract['lastPrice'].values[0])
            iv = float(contract['impliedVolatility'].values[0])
            days_to_exp = (datetime.strptime(expiration, "%Y-%m-%d") - datetime.now()).days
            T = max(days_to_exp / 365.0, 0.001)
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
            
            results[occ] = {'option_price': option_price, 'spot': spot, 'delta': delta, 'gamma': gamma}
        except Exception:
            results[occ] = {'option_price': None, 'spot': None, 'delta': None, 'gamma': None}
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

    strike_disp = int(strike) if strike == int(strike) else strike
    target_disp = int(target) if target == int(target) else target

    # Buy: want price to drop to target. Sell: want price to rise to target.
    if current_price is not None:
        hit = current_price <= target if intent == 'buy' else current_price >= target
    else:
        hit = False

    # Stock move needed: delta-gamma quadratic is more accurate than delta-only for large moves.
    # Solve: ½·gamma·dS² + delta·dS - dOption = 0
    # dS = (-delta ± sqrt(delta² + 2·gamma·dOption)) / gamma
    # Fall back to delta-only (dS = dOption/delta) when gamma is negligible.
    stock_target_str = "N/A"
    if spot is not None and delta is not None and delta != 0 and current_price is not None:
        d_option = target - current_price
        if gamma is not None and abs(gamma) > 1e-8:
            discriminant = delta ** 2 + 2 * gamma * d_option
            if discriminant >= 0:
                # Two roots — pick the one consistent with the direction delta implies
                sqrt_disc = np.sqrt(discriminant)
                ds1 = (-delta + sqrt_disc) / gamma
                ds2 = (-delta - sqrt_disc) / gamma
                # delta sign tells us which direction the stock should move
                ds = ds1 if (delta > 0 and ds1 > ds2) or (delta < 0 and ds1 < ds2) else ds2
            else:
                # No real solution (target unreachable with current IV/time); fall back
                ds = d_option / delta
        else:
            ds = d_option / delta
        stock_target_str = f"${spot + ds:,.2f}"

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

@st.cache_data(ttl=120)
def fetch_stock_prices(tickers):
    """Fetches price, day change, and technical indicators for a tuple of stock tickers."""
    results = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
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
            t = yf.Ticker(ticker)
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
            df = df.sort_values('Curr Value', key=lambda x: x.abs().fillna(0), ascending=False)

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

@st.cache_data(ttl=60)
def fetch_sentiment_prices(tickers):
    """Fetches current prices for a list of tickers from Yahoo Finance."""
    results = {}
    for ticker_id, ticker_info in tickers.items():
        try:
            ticker = yf.Ticker(ticker_id)
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
                "color": ticker_info["color"]
            }
        except Exception as e:
            results[ticker_id] = {"error": str(e), "name": ticker_info["name"]}
    return results

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
    }
    prices = fetch_sentiment_prices(tickers_to_fetch)

    sent_updated_str = datetime.now().strftime("%H:%M:%S")
    sent_status_color = '#22c55e' if sent_live_on else '#6B7280'
    sent_live_badge = (
        f'<span style="background:{sent_status_color};color:#000;font-size:0.65em;'
        f'padding:2px 8px;border-radius:10px;font-weight:700;">{"LIVE" if sent_live_on else "DELAYED"}</span>'
        f'&nbsp;<span style="color:#6B7280;font-size:0.72em;">Updated {sent_updated_str}</span>'
    )

    prices_html = '<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:12px;">'
    for ticker_id, data in prices.items():
        if "error" in data:
            price_html = f'<span style="color:#f87171;font-size:0.9em;">Error</span>'
        else:
            change_color = '#22c55e' if data['change'] >= 0 else '#f87171'
            arrow = '▲' if data['change'] >= 0 else '▼'
            price_html = (
                f'<span style="color:#e2e8f0;font-size:1.15em;font-weight:700;">${data["price"]:,.2f}</span>'
                f'<span style="color:{change_color};font-size:0.85em;font-weight:600;margin-left:8px;">'
                f'{arrow} {abs(data["change"]):,.2f} ({data["change_pct"]:+.2f}%)</span>'
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
        st.rerun()

    st.caption("Prices via Yahoo Finance. CNN/Crypto indices refresh every 30 min.")

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
        st.rerun()
