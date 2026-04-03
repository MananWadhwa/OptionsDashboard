import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import scipy.stats as si
import re
from datetime import datetime
import glob
import os
import urllib.request
import json
import time

st.set_page_config(page_title="Options Tracker", layout="wide")

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
            spot_price = _yf_fetch_with_retry(
                lambda t=underlying_ticker: t.history(period="1d")['Close'].iloc[-1]
            )

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

                    last_price = contract['lastPrice'].values[0]
                    iv         = contract['impliedVolatility'].values[0]

                    days_to_exp = (datetime.strptime(expiration, "%Y-%m-%d") - datetime.now()).days
                    T = max(days_to_exp / 365.0, 0.001)

                    delta, theta, _ = calculate_greeks(spot_price, strike, T, r, iv, opt_type)

                    results.append({
                        "OCC_Symbol":       occ,
                        "Underlying_Price": spot_price,
                        "Current_Price":    last_price,
                        "Delta":            delta,
                        "Theta":            theta,
                        "DTE":              days_to_exp
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

# --- MAIN DASHBOARD ---
st.markdown(
    '<div style="font-size:1.1em;font-weight:700;color:#e2e8f0;'
    'padding:4px 0 10px;letter-spacing:0.01em;">Options Tracker</div>',
    unsafe_allow_html=True
)
page_tab1, page_tab2, page_tab3 = st.tabs(["Portfolio", "Watchlist", "Sentiment"])

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

    # Account filter
    accounts = ["All"] + positions['Account'].unique().tolist()
    selected_accounts = st.multiselect("Filter by Account", options=accounts, default=["All"])

    if "All" in selected_accounts:
        filtered_positions = positions
    else:
        filtered_positions = positions[positions['Account'].isin(selected_accounts)]

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

            singles_df['Price_Diff_To_Target'] = np.where(
                singles_df['Side'].str.upper() == 'LONG',
                singles_df['Target_Price'] - singles_df['Current_Price'],
                singles_df['Current_Price'] - singles_df['Target_Price']
            )
            singles_df['Days_To_Target_(Theta)'] = np.where(
                singles_df['Theta'] != 0,
                singles_df['Price_Diff_To_Target'] / np.abs(singles_df['Theta']),
                np.nan
            )
            singles_df['Underlying_Move_Needed_$'] = np.where(
                singles_df['Delta'] != 0,
                singles_df['Price_Diff_To_Target'] / singles_df['Delta'],
                np.nan
            )
            singles_df['Underlying_Move_Needed_$'] = np.where(
                (singles_df['Side'].str.upper() == 'SHORT') & (singles_df['Delta'] != 0),
                -singles_df['Underlying_Move_Needed_$'],
                singles_df['Underlying_Move_Needed_$']
            )
            singles_df['Target_Hit'] = np.where(
                singles_df['Side'].str.upper() == 'LONG',
                singles_df['Current_Price'] >= singles_df['Target_Price'],
                singles_df['Current_Price'] <= singles_df['Target_Price']
            )
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

                spread_delta = 0.0
                spread_theta = 0.0
                for _, leg in group.iterrows():
                    sign = 1 if leg['Side'].upper() == 'SHORT' else -1
                    spread_delta += sign * leg['Delta']
                    spread_theta += sign * leg['Theta']

                spread_price_change_needed = spread_target - net_current_price
                days_to_target = spread_price_change_needed / spread_theta if spread_theta != 0 else np.nan
                underlying_move = spread_price_change_needed / spread_delta if spread_delta != 0 else np.nan
                target_hit = net_current_price <= spread_target if is_credit_spread else net_current_price >= spread_target

                aggregated_spreads.append({
                    'Account': group['Account'].iloc[0],
                    'OCC_Symbol': spread_name,
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
            'Account', 'OCC_Symbol', 'Side', 'Quantity', 'Entry_Price', 'Current_Price', 'Target_Price',
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

            badge = '<span style="background:#22c55e;color:#000;font-size:0.6em;padding:2px 7px;border-radius:4px;font-weight:bold;vertical-align:middle;">TARGET HIT</span>' if target_hit else ''

            entry_str   = f"${entry:.2f}"   if pd.notna(entry)   else "—"
            current_str = f"${current:.2f}" if pd.notna(current) else "—"
            target_str  = f"${target:.2f}"  if pd.notna(target)  else "—"
            pnl_str     = f"${pnl:+,.2f}"   if pd.notna(pnl)     else "—"
            pnl_pct_str = f"{pnl_pct:+.1f}%" if pd.notna(pnl_pct) and pnl_pct == pnl_pct else ""
            dte_str     = f"{dte:.0f}d"      if pd.notna(dte) and dte == dte else "—"
            move_str    = f"${move:+.2f}"    if pd.notna(move) and move == move else "—"

            return f"""
<div style="border:1.5px solid {border_color};border-radius:12px;padding:14px 16px;
            background:{bg_color};">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;">
    <div style="font-size:0.95em;font-weight:600;line-height:1.4;">{option_html}</div>
    <div style="text-align:right;flex-shrink:0;margin-left:8px;">
      <span style="color:{side_color};font-size:0.72em;font-weight:700;text-transform:uppercase;">{side}</span>
      <span style="color:#6B7280;font-size:0.72em;margin-left:4px;">×{qty}</span>
    </div>
  </div>
  {('<div style="margin-bottom:6px;">' + badge + '</div>') if target_hit else ''}
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px 8px;margin-top:8px;">
    <div>
      <div style="color:#6B7280;font-size:0.62em;text-transform:uppercase;">Entry</div>
      <div style="color:#9CA3AF;font-size:0.88em;font-weight:600;">{entry_str}</div>
    </div>
    <div>
      <div style="color:#6B7280;font-size:0.62em;text-transform:uppercase;">Current</div>
      <div style="color:#e2e8f0;font-size:0.88em;font-weight:600;">{current_str}</div>
    </div>
    <div>
      <div style="color:#6B7280;font-size:0.62em;text-transform:uppercase;">Target</div>
      <div style="color:#C084FC;font-size:1em;font-weight:700;">{target_str}</div>
    </div>
  </div>
  <div style="margin-top:10px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.07);
              display:grid;grid-template-columns:1fr 1fr;gap:4px 8px;">
    <div>
      <div style="color:#6B7280;font-size:0.62em;text-transform:uppercase;">Unreal. P&L</div>
      <div style="color:{pnl_color};font-size:1em;font-weight:700;">{pnl_str}
        <span style="font-size:0.7em;opacity:0.8;">{pnl_pct_str}</span>
      </div>
    </div>
    <div>
      <div style="color:#6B7280;font-size:0.62em;text-transform:uppercase;">Days → Target</div>
      <div style="color:#e2e8f0;font-size:0.88em;font-weight:600;">{dte_str}</div>
    </div>
    <div style="grid-column:1/-1;">
      <div style="color:#6B7280;font-size:0.62em;text-transform:uppercase;">Underlying Move Needed</div>
      <div style="color:#60A5FA;font-size:0.88em;font-weight:600;">{move_str}</div>
    </div>
  </div>
</div>"""

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

        st.caption("Data is delayed. Greeks are Black-Scholes approximations.")

        if st.button("Refresh Market Data"):
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
WATCHLIST_COLS = ['Ticker', 'ExpirationYYMMDD', 'OptionType', 'Strike', 'TargetPrice', 'Intent', 'Label']

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
            spot = float(underlying_ticker.history(period="1d")['Close'].iloc[-1])
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

with page_tab2:
    watchlist = load_watchlist()

    # Fetch prices
    if not watchlist.empty:
        occ_list = [watchlist_occ(r) for _, r in watchlist.iterrows()]
        with st.spinner("Fetching watchlist prices..."):
            prices = fetch_watchlist_prices(tuple(occ_list))

        # Display cards
        cards_html = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:8px;width:100%;">'
        for i, (_, row) in enumerate(watchlist.iterrows()):
            occ = occ_list[i]
            cards_html += watchlist_card_html(row, prices.get(occ))
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        if st.button("Refresh Watchlist", key="wl_refresh"):
            fetch_watchlist_prices.clear()
            st.rerun()
    else:
        st.info("No options in your watchlist. Add one below.")

    st.divider()
    with st.expander("Manage Watchlist", expanded=watchlist.empty):
        wl_add, wl_edit, wl_delete = st.tabs(["Add", "Edit", "Delete"])

        with wl_add:
            with st.form("wl_add_form"):
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                wl_ticker  = c1.text_input("Ticker", placeholder="NVDA").strip().upper()
                wl_exp     = c2.text_input("Expiry YYMMDD", placeholder="270115")
                wl_type    = c3.selectbox("Type", ["P", "C"])
                wl_strike  = c4.number_input("Strike", min_value=0.01, step=0.5, value=100.0)
                wl_target  = c5.number_input("Target Price", min_value=0.0, step=0.01, value=0.0)
                wl_intent  = c6.selectbox("Intent", ["Buy", "Sell"])
                wl_label   = st.text_input("Label (optional)", placeholder="e.g. earnings play")

                if st.form_submit_button("Add to Watchlist"):
                    if not wl_ticker or not wl_exp:
                        st.error("Ticker and Expiry are required.")
                    else:
                        new_wl = pd.DataFrame([{
                            'Ticker': wl_ticker, 'ExpirationYYMMDD': int(wl_exp),
                            'OptionType': wl_type, 'Strike': float(wl_strike),
                            'TargetPrice': float(wl_target),
                            'Intent': wl_intent.lower(),
                            'Label': wl_label.strip()
                        }])
                        save_watchlist(pd.concat([watchlist, new_wl], ignore_index=True))
                        fetch_watchlist_prices.clear()
                        st.success(f"Added {wl_ticker} {wl_type} ${wl_strike} to watchlist.")
                        st.rerun()

        with wl_edit:
            if watchlist.empty:
                st.info("No items to edit.")
            else:
                wl_labels = [f"{r['Ticker']} {r['OptionType']} {r['ExpirationYYMMDD']} ${r['Strike']}" for _, r in watchlist.iterrows()]
                wl_sel     = st.selectbox("Select item", wl_labels, key="wl_edit_sel")
                wl_sel_idx = wl_labels.index(wl_sel)
                wl_row     = watchlist.iloc[wl_sel_idx]

                with st.form("wl_edit_form"):
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
                wl_del_labels = [f"{r['Ticker']} {r['OptionType']} {r['ExpirationYYMMDD']} ${r['Strike']}" for _, r in watchlist.iterrows()]
                wl_del_sel    = st.selectbox("Select item to remove", wl_del_labels, key="wl_del_sel")
                wl_del_idx    = wl_del_labels.index(wl_del_sel)

                st.warning(f"Remove **{wl_del_sel}** from watchlist?")
                if st.button("Remove", type="primary", key="wl_del_btn"):
                    save_watchlist(watchlist.drop(index=wl_del_idx).reset_index(drop=True))
                    fetch_watchlist_prices.clear()
                    st.success("Removed from watchlist.")
                    st.rerun()
# =====================================================================
# --- SENTIMENT TAB ---
# =====================================================================

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
        st.rerun()

    st.caption("CNN index refreshes every 30 min. Crypto index via alternative.me.")
