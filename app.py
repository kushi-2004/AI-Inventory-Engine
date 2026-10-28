import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import plotly.express as px

st.set_page_config(
    page_title="AI Inventory Optimization Engine",
    layout="wide"
)

bg_image_url = "https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?q=80&w=2070&auto=format&fit=crop"

st.markdown(f"""
    <style>
    /* Background Image with Dark Overlay */
    .stApp {{
        background: linear-gradient(rgba(9, 13, 22, 0.85), rgba(9, 13, 22, 0.90)), url("{bg_image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* Centered Hero Section Styling */
    .hero-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 40px 20px;
        width: 100%;
    }}
    .main-title {{
        background: linear-gradient(90deg, #60a5fa 0%, #34d399 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 3.5rem;
        letter-spacing: -1px;
        margin-bottom: 15px;
        text-shadow: 0 20px 40px rgba(0,0,0,0.5);
        text-align: center;
    }}
    .hero-subtitle {{
        color: #94a3b8;
        font-size: 1.25rem;
        font-weight: 400;
        max-width: 900px;
        margin: 0 auto 25px auto;
        text-align: center;
    }}
    
    /* Animated Scroll Down Button Styling */
    .scroll-btn {{
        background: rgba(96, 165, 250, 0.15);
        border: 1px solid rgba(96, 165, 250, 0.4);
        color: #60a5fa;
        padding: 12px 28px;
        border-radius: 30px;
        text-decoration: none;
        font-weight: 600;
        font-size: 1rem;
        display: inline-block;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        animation: bounce 2s infinite;
    }}
    .scroll-btn:hover {{
        background: rgba(96, 165, 250, 0.3);
        border-color: #60a5fa;
        color: #ffffff;
        transform: translateY(-2px);
    }}
    @keyframes bounce {{
        0%, 20%, 50%, 80%, 100% {{ transform: translateY(0); }}
        40% {{ transform: translateY(-6px); }}
        60% {{ transform: translateY(-3px); }}
    }}

    /* Glassmorphism Metric Cards */
    .metric-card {{
        background: rgba(17, 24, 39, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        text-align: center;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-5px);
        border-color: #60a5fa;
    }}
    
    /* Transparent DataFrames & Plots Background */
    .stDataFrame {{
        background: rgba(17, 24, 39, 0.5) !important;
        border-radius: 12px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }}
    
    /* Sidebar Transparent Styling */
    [data-testid="stSidebar"] {{
        background-color: rgba(11, 17, 32, 0.85) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="hero-container">
        <h1 class="main-title">AI-Driven Inventory & Reorder Engine</h1>
        <p class="hero-subtitle">Next-Gen Machine Learning Forecasting, Dynamic Safety Stock, and Automated PO Procurement</p>
        <a href="#metrics-section" class="scroll-btn">👇 Explore Dashboard</a>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div style="height: 520px;"></div>', unsafe_allow_html=True)

st.markdown('<div id="metrics-section"></div>', unsafe_allow_html=True)

@st.cache_resource
def train_inventory_ml_model():
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", end="2026-08-01", freq="D")
    skus_info = {
        "SKU-1001": {"supplier": "Alpha Supplies", "cost": 25.0, "stock": 12, "base": 18, "lead": 7},
        "SKU-1002": {"supplier": "Global Tech", "cost": 10.0, "stock": 140, "base": 8, "lead": 5},
        "SKU-1003": {"supplier": "Apex Wholesalers", "cost": 50.0, "stock": 5, "base": 25, "lead": 10},
        "SKU-1004": {"supplier": "Beta Logistics", "cost": 15.0, "stock": 45, "base": 12, "lead": 7}
    }
    
    all_data = []
    for sku, info in skus_info.items():
        sales = info["base"] + np.random.poisson(lam=4, size=len(dates))
        temp_df = pd.DataFrame({"date": dates, "sku_id": sku, "quantity_sold": sales})
        all_data.append(temp_df)
        
    df_full = pd.concat(all_data)
    
    df_full['dayofweek'] = df_full['date'].dt.dayofweek
    df_full['month'] = df_full['date'].dt.month
    df_full['lag_1'] = df_full.groupby('sku_id')['quantity_sold'].shift(1)
    df_full['lag_7'] = df_full.groupby('sku_id')['quantity_sold'].shift(7)
    df_full['rolling_mean_7'] = df_full.groupby('sku_id')['quantity_sold'].transform(lambda x: x.shift(1).rolling(7).mean())
    df_full['rolling_std_7'] = df_full.groupby('sku_id')['quantity_sold'].transform(lambda x: x.shift(1).rolling(7).std())
    
    df_full = df_full.dropna()
    
    features = ['dayofweek', 'month', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7']
    X = df_full[features]
    y = df_full['quantity_sold']
    
    model = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
    model.fit(X, y)
    
    return model, df_full, skus_info

model, df_historical, skus_info = train_inventory_ml_model()

st.sidebar.header("Simulation Controls")
st.sidebar.markdown("Fine-tune supply chain parameters.")
global_lead_time_override = st.sidebar.slider("Override Lead Time (Days)", 1, 30, 7)
service_level = st.sidebar.selectbox("Service Level Target", [0.90, 0.95, 0.99], index=1)
z_dict = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
z_score = z_dict[service_level]

st.sidebar.markdown("---")
st.sidebar.info("**Pro Tip:** Slide the lead time to instantly observe automated Safety Stock and Reorder Point recalculations.")

inventory_results = []
future_dates = pd.date_range(start="2026-08-02", periods=14, freq="D")
predictions_dict = {}

for sku, info in skus_info.items():
    sku_hist = df_historical[df_historical['sku_id'] == sku].tail(10)
    recent_row = sku_hist.iloc[-1]
    pred_sales_list = []
    
    curr_lag_1 = recent_row['quantity_sold']
    curr_lag_7 = sku_hist.iloc[-7]['quantity_sold'] if len(sku_hist) >= 7 else curr_lag_1
    curr_roll_mean = sku_hist['quantity_sold'].tail(7).mean()
    curr_roll_std = sku_hist['quantity_sold'].tail(7).std()
    if pd.isna(curr_roll_std): curr_roll_std = 2.0
    
    for f_date in future_dates:
        feat_df = pd.DataFrame([{
            'dayofweek': f_date.dayofweek,
            'month': f_date.month,
            'lag_1': curr_lag_1,
            'lag_7': curr_lag_7,
            'rolling_mean_7': curr_roll_mean,
            'rolling_std_7': curr_roll_std
        }])
        pred = model.predict(feat_df)[0]
        pred = max(1, int(round(pred)))
        pred_sales_list.append(pred)
        
        curr_lag_7 = curr_lag_1
        curr_lag_1 = pred
        
    avg_pred_demand = np.mean(pred_sales_list)
    std_pred_demand = np.std(pred_sales_list)
    if std_pred_demand == 0: std_pred_demand = 1.5
    
    predictions_dict[sku] = {
        "dates": future_dates,
        "preds": pred_sales_list
    }
    
    lt = global_lead_time_override
    safety_stock = int(np.ceil(z_score * std_pred_demand * np.sqrt(lt)))
    reorder_point = int(np.ceil((avg_pred_demand * lt) + safety_stock))
    
    current_stock = info["stock"]
    if current_stock <= reorder_point:
        status = "Reorder Needed"
        order_qty = max(0, (reorder_point * 2) - current_stock)
    else:
        status = "Sufficient Stock"
        order_qty = 0
        
    inventory_results.append({
        "SKU": sku,
        "Supplier": info["supplier"],
        "Current Stock": current_stock,
        "ML Predicted Demand (Daily)": round(avg_pred_demand, 2),
        "Lead Time": lt,
        "Safety Stock": safety_stock,
        "ROP": reorder_point,
        "Status": status,
        "Order Qty": order_qty,
        "Total PO Cost ($)": order_qty * info["cost"]
    })

res_df = pd.DataFrame(inventory_results)


st.markdown("---")
st.subheader("Executive Summary Metrics")

total_skus = len(res_df)
low_stock_count = len(res_df[res_df['Status'].str.contains("Reorder")])
total_po_value = res_df['Total PO Cost ($)'].sum()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <p style='color: #9ca3af; font-size: 14px; font-weight: 500; margin-bottom: 4px;'>TOTAL SKUS MANAGED</p>
            <h2 style='color: #60a5fa; font-size: 2.2rem; margin: 0; font-weight: 700;'>{total_skus}</h2>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card">
            <p style='color: #9ca3af; font-size: 14px; font-weight: 500; margin-bottom: 4px;'>LOW STOCK ALERTS</p>
            <h2 style='color: #f87171; font-size: 2.2rem; margin: 0; font-weight: 700;'>{low_stock_count}</h2>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card">
            <p style='color: #9ca3af; font-size: 14px; font-weight: 500; margin-bottom: 4px;'>TOTAL DRAFT PO VALUE</p>
            <h2 style='color: #34d399; font-size: 2.2rem; margin: 0; font-weight: 700;'>${total_po_value:,.2f}</h2>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.subheader("AI Demand Forecasting Visualizer")
selected_sku = st.selectbox("Select SKU for Deep-Dive ML Trend Analysis", list(skus_info.keys()))

hist_sku_df = df_historical[df_historical['sku_id'] == selected_sku].tail(30)[['date', 'quantity_sold']]
hist_sku_df['Type'] = 'Historical Sales'
hist_sku_df.columns = ['Date', 'Sales', 'Type']

fc_sku_df = pd.DataFrame({
    'Date': predictions_dict[selected_sku]['dates'],
    'Sales': predictions_dict[selected_sku]['preds'],
    'Type': 'LightGBM Forecast'
})

df_chart = pd.concat([hist_sku_df, fc_sku_df])

fig = px.line(
    df_chart, 
    x="Date", 
    y="Sales", 
    color="Type", 
    markers=True,
    color_discrete_map={'Historical Sales': '#3b82f6', 'LightGBM Forecast': '#34d399'}
)
fig.update_layout(
    template="plotly_dark", 
    xaxis_title="Date", 
    yaxis_title="Quantity Sold",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color="#f3f4f6", family="Inter"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.subheader("Inventory Status & Reorder Recommendations")
st.dataframe(res_df, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

st.subheader("Automated Draft Purchase Orders")
reorder_items = res_df[res_df['Order Qty'] > 0]

if not reorder_items.empty:
    st.success(f"Action Required: Found {len(reorder_items)} SKUs requiring immediate replenishment.")
    st.dataframe(reorder_items[["SKU", "Supplier", "Order Qty", "Total PO Cost ($)"]], use_container_width=True)
    
    csv_data = reorder_items.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Verified Purchase Orders (CSV)",
        data=csv_data,
        file_name="verified_purchase_orders.csv",
        mime="text/csv"
    )
else:
    st.info("Optimal Stock Levels: All items are currently well above safety thresholds.")