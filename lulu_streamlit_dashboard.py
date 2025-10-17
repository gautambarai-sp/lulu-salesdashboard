# lulu_streamlit_dashboard_altair.py
# Altair-based Streamlit dashboard for Lulu UAE sales analysis
# Usage:
#   pip install streamlit pandas altair scipy numpy
#   streamlit run lulu_streamlit_dashboard_altair.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Lulu UAE — Altair Dashboard", layout="wide")

# Enable Altair data transformer for larger datasets
alt.data_transformers.enable('default', max_rows=50000)

# ----------------- Helpers -----------------
def load_csv_safe(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        try:
            return pd.read_csv(p, engine="python")
        except Exception:
            return None

def detect_columns(df):
    cols = df.columns.tolist()
    def find(terms):
        for t in terms:
            for c in cols:
                if t in c.lower():
                    return c
        return None
    return {
        'amount': find(['amount','sales','revenue','net','total','paid','value']),
        'qty': find(['qty','quantity','units']),
        'department': find(['department','dept']),
        'store_format': find(['store_format','store format','format','storetype','store_type']),
        'category': find(['category','cat','sub_category','subcat']),
        'product': find(['product','sku','item','product_name']),
        'campaign': find(['campaign','ad_campaign']),
        'promo': find(['promo','voucher','coupon','promo_code']),
        'gender': find(['gender']),
        'nationality': find(['national','country','nationality']),
        'city': find(['city','location']),
        'transaction': find(['invoice','transaction','order','receipt','bill','txn']),
        'date': find(['date'])
    }

def prepare_dataframe(df, mapping):
    d = df.copy()
    ren = {}
    if mapping.get('amount'): ren[mapping['amount']] = 'SalesAmount'
    if mapping.get('qty'): ren[mapping['qty']] = 'Quantity'
    if mapping.get('department'): ren[mapping['department']] = 'Department'
    if mapping.get('store_format'): ren[mapping['store_format']] = 'Store_format'
    if mapping.get('category'): ren[mapping['category']] = 'Category'
    if mapping.get('product'): ren[mapping['product']] = 'Product'
    if mapping.get('campaign'): ren[mapping['campaign']] = 'Campaign'
    if mapping.get('promo'): ren[mapping['promo']] = 'PromoCode'
    if mapping.get('gender'): ren[mapping['gender']] = 'Gender'
    if mapping.get('nationality'): ren[mapping['nationality']] = 'Nationality'
    if mapping.get('city'): ren[mapping['city']] = 'City'
    if mapping.get('transaction'): ren[mapping['transaction']] = 'Transaction'
    if mapping.get('date'): ren[mapping['date']] = 'Date'
    d = d.rename(columns=ren)

    if 'SalesAmount' in d.columns:
        d['SalesAmount'] = pd.to_numeric(d['SalesAmount'], errors='coerce').fillna(0.0)
    else:
        d['SalesAmount'] = 1.0
    if 'Quantity' in d.columns:
        d['Quantity'] = pd.to_numeric(d['Quantity'], errors='coerce').fillna(1)
    else:
        d['Quantity'] = 1
    if 'Transaction' not in d.columns:
        d['Transaction'] = d.index.astype(str)
    else:
        d['Transaction'] = d['Transaction'].astype(str)

    text_cols = ['Department','Store_format','Category','Product','Campaign','PromoCode','Gender','Nationality','City']
    for c in text_cols:
        if c in d.columns:
            d[c] = d[c].fillna('Unknown').astype(str)

    if 'Date' in d.columns:
        d['Date'] = pd.to_datetime(d['Date'], errors='coerce')

    if 'PromoCode' in d.columns:
        d['PromoUsed'] = d['PromoCode'].astype(str).str.strip().replace({'nan':'','None':''}).apply(lambda x: bool(x) and x!='')
    else:
        d['PromoUsed'] = False

    return d

def top_n(df, col, n=10):
    if col not in df.columns:
        return pd.DataFrame()
    agg = df.groupby(col).agg(TotalSales=('SalesAmount','sum'), Transactions=('Transaction','nunique')).reset_index()
    agg['AvgBasket'] = agg['TotalSales'] / agg['Transactions'].replace(0, np.nan)
    return agg.sort_values('TotalSales', ascending=False).head(n)

def promo_effectiveness(df, group_cols):
    results = []
    grouped = df.groupby(group_cols)
    for name, g in grouped:
        key = name if isinstance(name, str) else ' | '.join([str(x) for x in name])
        arr_with = g[g['PromoUsed']].groupby('Transaction')['SalesAmount'].sum().values
        arr_without = g[~g['PromoUsed']].groupby('Transaction')['SalesAmount'].sum().values
        sales_with = arr_with.sum() if arr_with.size > 0 else 0
        sales_without = arr_without.sum() if arr_without.size > 0 else 0
        tx_with = len(np.unique(g[g['PromoUsed']]['Transaction'])) if g[g['PromoUsed']].shape[0] > 0 else 0
        tx_without = len(np.unique(g[~g['PromoUsed']]['Transaction'])) if g[~g['PromoUsed']].shape[0] > 0 else 0
        avg_with = arr_with.mean() if arr_with.size > 0 else 0
        avg_without = arr_without.mean() if arr_without.size > 0 else 0
        uplift = np.nan
        if avg_without > 0:
            uplift = (avg_with - avg_without) / avg_without * 100
        pval = np.nan
        results.append({
            'Group': key,
            'SalesWithPromo': sales_with,
            'SalesWithoutPromo': sales_without,
            'TxWithPromo': tx_with,
            'TxWithoutPromo': tx_without,
            'AvgWith': avg_with,
            'AvgWithout': avg_without,
            'Uplift_pct': uplift,
            'pval': pval
        })
    return pd.DataFrame(results).sort_values('SalesWithPromo', ascending=False)

# ----------------- Load data or upload -----------------
DEFAULT_PATH = "/mnt/data/lulu_uae_master_2000.csv"
raw = load_csv_safe(DEFAULT_PATH)

st.sidebar.header("Data options")
use_upload = st.sidebar.checkbox("Upload CSV instead of using default file", value=False)
if use_upload:
    uploaded = st.sidebar.file_uploader("Upload transactions CSV", type=["csv"])
    if uploaded is not None:
        raw = pd.read_csv(uploaded)

if raw is None:
    st.error(f"Could not load CSV at {DEFAULT_PATH}. Please upload a file using the sidebar.")
    st.stop()

mapping = detect_columns(raw)
df = prepare_dataframe(raw, mapping)

# ----------------- Filters -----------------
st.sidebar.header("Filters")
if 'Date' in df.columns and df['Date'].notna().any():
    min_d = df['Date'].min().date()
    max_d = df['Date'].max().date()
    date_range = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
else:
    date_range = None

city_values = ['All'] + sorted(df['City'].unique().tolist()) if 'City' in df.columns else ['All']
city = st.sidebar.selectbox("City", city_values)

store_values = ['All'] + sorted(df['Store_format'].unique().tolist()) if 'Store_format' in df.columns else ['All']
store = st.sidebar.selectbox("Store Type", store_values)

mask = pd.Series(True, index=df.index)
if date_range and 'Date' in df.columns:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    mask &= (df['Date'] >= start) & (df['Date'] <= end)
if city != 'All' and 'City' in df.columns:
    mask &= (df['City'] == city)
if store != 'All' and 'Store_format' in df.columns:
    mask &= (df['Store_format'] == store)

filtered = df[mask].copy()

# ----------------- KPIs -----------------
st.title("Lulu UAE — Altair Dashboard")
st.write("Flow: City → Store Type → Department → Category/Product → Promo Effectiveness → Gender/Nationality")

col1, col2, col3, col4 = st.columns(4)
total_sales = filtered['SalesAmount'].sum()
total_tx = filtered['Transaction'].nunique()
avg_basket = filtered.groupby('Transaction')['SalesAmount'].sum().mean() if total_tx > 0 else 0
total_qty = filtered['Quantity'].sum()
col1.metric("Total Sales", f"{total_sales:,.0f}")
col2.metric("Transactions", f"{total_tx:,}")
col3.metric("Avg Basket", f"{avg_basket:,.2f}")
col4.metric("Total Qty", f"{total_qty:,}")

# ----------------- City analysis -----------------
st.header("1) City Analysis")
if 'City' in filtered.columns:
    city_df = filtered.groupby('City').agg(TotalSales=('SalesAmount', 'sum')).reset_index().sort_values('TotalSales', ascending=False)
    chart = alt.Chart(city_df).mark_bar().encode(
        x=alt.X('City:N', sort='-y'),
        y='TotalSales:Q',
        color='TotalSales:Q'
    ).properties(width=800, height=400).interactive()
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(city_df, use_container_width=True)
else:
    st.info("No City column.")

# ----------------- Store type -----------------
st.header("2) Store Type")
if 'Store_format' in filtered.columns:
    sf = filtered.groupby('Store_format').agg(TotalSales=('SalesAmount', 'sum')).reset_index().sort_values('TotalSales', ascending=False)
    chart = alt.Chart(sf).mark_bar().encode(
        x=alt.X('Store_format:N', sort='-y'),
        y='TotalSales:Q',
        color='TotalSales:Q'
    ).properties(width=800, height=400).interactive()
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(sf, use_container_width=True)
else:
    st.info("No Store_format column.")

# ----------------- Department / category / product -----------------
st.header("3) Department → Category → Product")
if 'Department' in filtered.columns:
    dept_choice = st.selectbox("Department", ['All'] + sorted(filtered['Department'].unique().tolist()))
    dept_df = filtered if dept_choice == 'All' else filtered[filtered['Department'] == dept_choice]
    dept_top = top_n(dept_df, 'Department', 50)
    st.subheader("Department summary")
    st.dataframe(dept_top, use_container_width=True)
    
    if 'Category' in dept_df.columns:
        cat_top = top_n(dept_df, 'Category', 50)
        chart = alt.Chart(cat_top).mark_bar().encode(
            x=alt.X('Category:N', sort='-y'),
            y='TotalSales:Q',
            color='TotalSales:Q'
        ).properties(width=800, height=400).interactive()
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(cat_top, use_container_width=True)
    
    if 'Product' in dept_df.columns:
        prod_top = top_n(dept_df, 'Product', 200)
        st.subheader("Top products")
        st.dataframe(prod_top, use_container_width=True)
else:
    st.info("No Department column.")

# ----------------- Promo effectiveness -----------------
st.header("4) Promo & Discount Effectiveness")
levels = st.multiselect("Group by (choose level(s))", ['Department', 'Category', 'Product'], default=['Department', 'Category'])
valid_levels = [l for l in levels if l in filtered.columns]
if len(valid_levels) == 0:
    st.info("Pick levels present in data.")
else:
    with st.spinner("Calculating..."):
        promo_res = promo_effectiveness(filtered, valid_levels)
        st.subheader("Promo results")
        st.dataframe(promo_res, use_container_width=True)
        
        def decision(r):
            support = r['TxWithPromo'] + r['TxWithoutPromo']
            if support < 20:
                return "Insufficient data"
            if not np.isnan(r['Uplift_pct']) and r['Uplift_pct'] > 10 and (np.isnan(r['pval']) or r['pval'] < 0.1):
                return "Continue & scale"
            if not np.isnan(r['Uplift_pct']) and abs(r['Uplift_pct']) < 5 and r['TxWithPromo'] <= r['TxWithoutPromo']:
                return "Discontinue"
            return "Review"
        
        promo_res['Decision'] = promo_res.apply(decision, axis=1)
        st.subheader("Decisions (heuristic)")
        decision_df = promo_res[['Group', 'Uplift_pct', 'TxWithPromo', 'TxWithoutPromo', 'pval', 'Decision']]
        st.dataframe(decision_df, use_container_width=True)
        
        # Visualization of uplift
        chart = alt.Chart(promo_res).mark_bar().encode(
            x=alt.X('Group:N', sort='-y'),
            y='Uplift_pct:Q',
            color=alt.condition(
                alt.datum.Uplift_pct > 0,
                alt.value('green'),
                alt.value('red')
            )
        ).properties(width=800, height=400).interactive()
        st.altair_chart(chart, use_container_width=True)

# ----------------- Gender & Nationality -----------------
st.header("5) Gender & Nationality")
seg = st.selectbox("Segment level", ['Department', 'Category', 'Product'])
if seg in filtered.columns:
    seg_gender = filtered.groupby([seg, 'Gender']).agg(TotalSales=('SalesAmount', 'sum')).reset_index().sort_values('TotalSales', ascending=False)
    st.subheader("Sales by Gender")
    chart_gender = alt.Chart(seg_gender.head(50)).mark_bar().encode(
        x=alt.X('Gender:N'),
        y='TotalSales:Q',
        color='Gender:N',
        column=alt.Column(seg + ':N', wrap=3)
    ).properties(width=250, height=300).interactive()
    st.altair_chart(chart_gender, use_container_width=True)
    st.dataframe(seg_gender.head(200), use_container_width=True)
    
    seg_nat = filtered.groupby([seg, 'Nationality']).agg(TotalSales=('SalesAmount', 'sum')).reset_index().sort_values('TotalSales', ascending=False)
    st.subheader("Sales by Nationality")
    chart_nat = alt.Chart(seg_nat.head(50)).mark_bar().encode(
        x=alt.X('Nationality:N'),
        y='TotalSales:Q',
        color='Nationality:N'
    ).properties(width=800, height=400).interactive()
    st.altair_chart(chart_nat, use_container_width=True)
    st.dataframe(seg_nat.head(200), use_container_width=True)
else:
    st.info(f"{seg} not present in data.")

# ----------------- Decision matrix -----------------
st.header("6) City × Department Decision Matrix")
if 'City' in filtered.columns and 'Department' in filtered.columns:
    matrix = filtered.groupby(['City', 'Department']).agg(TotalSales=('SalesAmount', 'sum'), Transactions=('Transaction', 'nunique')).reset_index()
    matrix_sorted = matrix.sort_values('TotalSales', ascending=False)
    
    heatmap = alt.Chart(matrix_sorted).mark_rect().encode(
        x='City:N',
        y='Department:N',
        color='TotalSales:Q'
    ).properties(width=800, height=400).interactive()
    st.altair_chart(heatmap, use_container_width=True)
    st.dataframe(matrix_sorted.head(200), use_container_width=True)
else:
    st.info("No City or Department.")

# ----------------- Export -----------------
st.sidebar.header("Export")
if st.sidebar.button("Download filtered CSV"):
    csv_data = filtered.to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="filtered_transactions.csv",
        mime="text/csv"
    )