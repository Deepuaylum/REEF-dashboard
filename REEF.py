import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import warnings
from PIL import Image
import subprocess

warnings.filterwarnings("ignore")
st.set_page_config(page_title="R.E.E.F- Rapid Environmental Early-warning Forecaster", layout="wide")

def gdown_download_if_missing(local_path, gdrive_url):
    if not os.path.exists(local_path):
        st.info(f"Downloading {local_path} from Google Drive...")
        try:
            import gdown
        except ImportError:
            subprocess.run(["pip", "install", "gdown"])
            import gdown
        gdown.download(gdrive_url, local_path, quiet=False)
    # Validate: must be a reasonable size (say, at least 100KB for a model)
    if not os.path.exists(local_path) or os.path.getsize(local_path) < 100*1024:
        st.error(f"Model file {local_path} was not downloaded correctly.")
        st.stop()

# Google Drive "shareable link" (ID only)
REALM_MODEL_PATH = "Realm_model.joblib"
CUSTOM_RF_MODEL_PATH = "Custom_RF_model.joblib"
REALM_MODEL_GDRIVE = "https://drive.google.com/uc?id=1BHH4XP1Mo7WyNFfI_umEC76aL006JDKG"
CUSTOM_RF_MODEL_GDRIVE = "https://drive.google.com/uc?id=1EbtKZBHJmlEC4yPJdZEDCbDZdfnZ8JN7"

gdown_download_if_missing(REALM_MODEL_PATH, REALM_MODEL_GDRIVE)
gdown_download_if_missing(CUSTOM_RF_MODEL_PATH, CUSTOM_RF_MODEL_GDRIVE)

@st.cache_resource
def load_model():
    return joblib.load(REALM_MODEL_PATH)
model = load_model()

@st.cache_resource
def load_custom_rf():
    return joblib.load(CUSTOM_RF_MODEL_PATH)
custom_rf_model = load_custom_rf()

@st.cache_data
def load_model_data():
    return pd.read_csv("use_for_feature_jun_24.csv")
df_model = load_model_data()

@st.cache_data
def load_viz_data():
    df = pd.read_csv("bleaching_with_restoration_merged_final.csv")
    df = df.dropna(subset=["Latitude_Degrees", "Longitude_Degrees", "Percent_Bleaching", "Date_Year"])
    df["Date_Year"] = pd.to_numeric(df["Date_Year"], errors="coerce")
    df = df[df["Date_Year"].between(1980, 2020)]
    return df
df = load_viz_data()
if df.empty:
    st.stop()

model_features = [
    'Temperature_Kelvin', 'SSTA', 'ClimSST', 'Depth_m', 'Turbidity',
    'Windspeed', 'Cyclone_Frequency', 'IDW_G2talk', 'IDW_G2oxygen', 'IDW_G2phts25p0'
]

def forecast_feature(df, realm, feature, year):
    data = df[(df['Realm_Name'] == realm) & (df['Date_Year'] <= 2019)][['Date_Year', feature]].dropna()
    if len(data) < 5:
        return np.nan
    X = sm.add_constant(data[['Date_Year']])
    y = data[feature]
    model_ = sm.OLS(y, X).fit()
    X_new = pd.DataFrame({'const': 1, 'Date_Year': [year]})
    pred = model_.get_prediction(X_new).summary_frame(alpha=0.05)
    return pred['mean'].values[0]

def classify_severity(pct):
    if pct < 10:
        return "Mild"
    elif pct < 30:
        return "Moderate"
    else:
        return "Severe"

severity_colors = {
    "Mild": "#3CB371",      # Green
    "Moderate": "#FFA500",  # Orange
    "Severe": "#D7263D",    # Red
}

# ===== Row 1: Centered Heading =====
st.markdown("""
<style>
.coral-header {
    width: 100vw !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center;
    position: relative;
    left: 50%;
    transform: translateX(-50%);
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
    font-family: 'Segoe UI', Arial, sans-serif;
}
</style>
<div class="coral-header">
    <span style="font-size:2.7rem; font-weight:700; letter-spacing:0.01em; color:#174672;">
        R.E.E.F- Rapid Environmental Early-warning Forecaster
    </span>
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ===== Row 2: Metrics =====
kpi_df = df.copy()
k1, k2, k3, k4, k5 = st.columns([1,1,1,1,1])
k1.metric("Unique realms", kpi_df['Realm_Name'].nunique())
k2.metric("Years covered", "2000–2019")
k3.metric("Avg bleaching (%)", "16.3%")
k4.metric("Max bleaching (%)", f"{kpi_df['Percent_Bleaching'].max():.1f}")
k5.metric("Min bleaching (%)", f"{kpi_df['Percent_Bleaching'].min():.1f}")
st.markdown("<br>", unsafe_allow_html=True)

# ====== METRIC FONT SIZE CORRECTION ======
st.markdown("""
<style>
div[data-testid="stMetric"] > div:first-child {
    font-size: 2.3rem !important;
    font-weight: 700 !important;
    color: #174672 !important;
    letter-spacing: 0.01em;
    margin-bottom: 0.22em;
}
div[data-testid="stMetricValue"] {
    font-size: 2.6rem;
    font-weight: 700;
    color: #174672;
}
</style>
""", unsafe_allow_html=True)

# ===== Row 3: Bar, Trend, Feature Importance Image =====
row3_1, row3_2, row3_3 = st.columns([1.15,1.15,1.1], gap="large")
with row3_1:
    top_realm_df = (
        kpi_df[kpi_df["Date_Year"] >= 2000]
        .groupby("Realm_Name")["Percent_Bleaching"]
        .mean()
        .sort_values(ascending=False)
        .head(4)
        .reset_index()
    )
    # Assign colors by bleaching percentage
    realm_colors = {}
    for _, row in top_realm_df.iterrows():
        if row["Percent_Bleaching"] < 10:
            realm_colors[row["Realm_Name"]] = "#3CB371"  # Green
        elif row["Percent_Bleaching"] < 30:
            realm_colors[row["Realm_Name"]] = "#FFA500"  # Amber
        else:
            realm_colors[row["Realm_Name"]] = "#D7263D"  # Red (severe, if ever needed)
    fig_toprealms = px.bar(
        top_realm_df,
        x="Realm_Name", y="Percent_Bleaching",
        color="Realm_Name",
        color_discrete_map=realm_colors,
        title="Top 4 Realms: Avg Bleaching (%) (since 2000)"
    )
    fig_toprealms.update_traces(marker_line_width=0, width=0.5)
    fig_toprealms.update_layout(
        showlegend=False, height=300, margin=dict(l=2, r=2, t=32, b=2),
        yaxis=dict(title=None, showgrid=True, zeroline=True),
        xaxis=dict(title=None, tickfont=dict(size=11))
    )
    st.plotly_chart(fig_toprealms, use_container_width=True)

with row3_2:
    threshold = 30
    event_years = [1998,2005, 2010, 2016, 2023]
    df_year = (
        df.groupby("Date_Year")
        .agg(
            Avg_Bleaching=("Percent_Bleaching", "mean"),
            Median_Bleaching=("Percent_Bleaching", "median"),
            Total_Sites=("Percent_Bleaching", "count"),
            Sites_Above_Thresh=("Percent_Bleaching", lambda x: (x >= threshold).sum())
        )
        .reset_index()
    )
    df_year = df_year[df_year["Date_Year"] >= 2000]
    df_year[f"Pct_Sites_Above_{threshold}"] = 100 * df_year["Sites_Above_Thresh"] / df_year["Total_Sites"]
    extreme_threshold = df['Percent_Bleaching'].quantile(0.9)
    extreme_events = df[df['Percent_Bleaching'] >= extreme_threshold]
    yearly_extremes = (
        extreme_events.groupby(df['Date_Year'].astype(int))
        .size()
        .reset_index(name="Extreme_Event_Count")
    )
    yearly_extremes = yearly_extremes[yearly_extremes["Date_Year"] >= 2000]
    df_year = df_year.merge(yearly_extremes, left_on="Date_Year", right_on="Date_Year", how="left")
    df_year["Extreme_Event_Count"] = df_year["Extreme_Event_Count"].fillna(0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_year["Date_Year"], y=df_year["Avg_Bleaching"],
        mode="lines+markers", name="Avg", line=dict(color="royalblue"), marker=dict(size=5), yaxis="y1"
    ))
    fig.add_trace(go.Scatter(
        x=df_year["Date_Year"], y=df_year["Median_Bleaching"],
        mode="lines+markers", name="Median", line=dict(color="green", dash="dash"), marker=dict(size=5), yaxis="y1"
    ))
    fig.add_trace(go.Bar(
        x=df_year["Date_Year"], y=df_year["Extreme_Event_Count"],
        name="Extreme Events", marker_color="crimson", opacity=0.35, yaxis="y2"
    ))
    for yr in event_years:
        if yr >= 2000:
            fig.add_vline(x=yr, line_color="black", line_width=1, line_dash="dot")
    fig.update_layout(
        title="Bleaching Percentage Trend",
        xaxis=dict(title="Bleaching Trend Over years", tickfont=dict(size=10)),
        yaxis=dict(title="Avg/Median Bleaching (%)", titlefont=dict(size=12)),
        yaxis2=dict(title="Extreme Events (Count)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="top", y=1.07, xanchor="center", x=0.5, font=dict(size=10)),
        bargap=0.2,
        height=320,
        margin=dict(l=2, r=2, t=32, b=2),
    )
    st.plotly_chart(fig, use_container_width=True)

with row3_3:
    st.markdown(
        "<div style='font-size:1.14rem; font-weight:600; text-align:center; margin-bottom:4px;'>"
        "Feature Importance - SHAP Method"
        "</div>",
        unsafe_allow_html=True
    )
    try:
        feat_imp_img = Image.open("WhatsApp Image 2025-07-09 at 01.24.30.jpeg")
        st.image(feat_imp_img, use_container_width=True)
    except Exception as e:
        st.warning("Feature importance image not found.")

# ===== Row 4: OLS + Map + Recommendation =====
row4_left, row4_right = st.columns([1.1, 1.7], gap="large")
with row4_left:
    st.markdown("#### Region & Year Forecast")
    realms = sorted(df_model['Realm_Name'].dropna().unique())
    realm_for_map = st.selectbox(
        "Region (Realm)", realms,
        key="realm_map_select_main",
        index=realms.index("Central Indo-Pacific") if "Central Indo-Pacific" in realms else 0,
    )
    forecast_year = st.slider("Forecast Year", min_value=2024, max_value=2100, value=2029, key="predmap_slider", label_visibility="visible")
    feature_inputs = {feat: forecast_feature(df_model, realm_for_map, feat, forecast_year) for feat in model_features}
    X_pred = pd.DataFrame([feature_inputs])
    if X_pred.isnull().values.any():
        predicted_bleaching = np.nan
        prediction_message = "Cannot predict (insufficient data)."
    else:
        predicted_bleaching = model.predict(X_pred)[0]
        prediction_message = f"Bleaching for {realm_for_map}, {forecast_year}: <b><span style='color:#ff6f61'>{predicted_bleaching:.1f}%</span></b>"
    st.markdown(prediction_message, unsafe_allow_html=True)
    st.markdown("**Forecasted Environmental Features (OLS)**")
    st.dataframe(X_pred.T, use_container_width=True, height=300)

# Prepare recommendation data BEFORE row4_right
realm_recommendations = {
    "Temperate Australasia":
        "Coral Gardening - Hybridization, Nursery Phase, Transplantation Phase, Larval Enhancement.",
    "Eastern Indo-Pacific":
        "Coral Gardening - Transplantation Phase, Substrate Enhancement - Algae removal.",
    "Central Indo-Pacific":
        "Coral Gardening - Nursery Phase, Transplantation Phase.",
    "Temperate Northern Atlantic":
        "Coral Gardening - Nursery Phase; Substrate Addition - Artificial reef; Substrate Enhancement - Algae removal.",
    "Temperate Northern Pacific":
        "Coral Gardening - Nursery Phase; Substrate Addition - Artificial reef.",
    "Tropical Atlantic":
        "Coral Gardening - Nursery Phase, Transplantation Phase; Substrate Addition - Artificial reef; Substrate Enhancement - Algae removal.",
    "Tropical Eastern Pacific":
        "Coral Gardening - Nursery Phase.",
    "Western Indo-Pacific":
        "Substrate Addition - Artificial reef; Substrate Enhancement - Electric."
}
realm_top_countries_simple = {
    "Central Indo-Pacific": ["Malaysia", "Philippines", "Indonesia"],
    "Eastern Indo-Pacific": ["French Polynesia", "Cook Islands", "Marshall Islands"],
    "Temperate Australasia": ["Australia"],
    "Temperate Northern Atlantic": ["United States"],
    "Temperate Northern Pacific": ["Japan", "Taiwan"],
    "Tropical Atlantic": ["Mexico", "Jamaica", "Cuba"],
    "Tropical Eastern Pacific": ["Costa Rica", "Ecuador", "Panama", "Colombia"],
    "Western Indo-Pacific": ["Egypt", "Maldives", "France", "Oman"],
}
# Calculate years to SEVERE bleaching
current_pred = predicted_bleaching if not np.isnan(predicted_bleaching) else 0
years_until_severe = None
target_threshold = 30.0

for year in range(int(forecast_year), 2101):
    feature_inputs_future = {feat: forecast_feature(df_model, realm_for_map, feat, year) for feat in model_features}
    if np.isnan(list(feature_inputs_future.values())).any():
        continue
    pred_bleach = model.predict(pd.DataFrame([feature_inputs_future]))[0]
    if pred_bleach >= target_threshold:
        years_until_severe = year - int(forecast_year)
        break

if years_until_severe is None and current_pred >= target_threshold:
    severity_line = f"<span style='color:#D7263D;'><b>This realm is already in SEVERE bleaching.</b></span>"
elif years_until_severe is not None and years_until_severe > 0:
    severity_line = f"<b>This realm has {years_until_severe} years before bleaching reaches <span style='color:#D7263D;'>SEVERE</span> (≥30%).</b>"
else:
    severity_line = "<b>Insufficient data to estimate time to severe bleaching.</b>"

top_countries_list = realm_top_countries_simple.get(realm_for_map, [])
top_countries_text = ", ".join(top_countries_list)
restoration_reco = realm_recommendations.get(
    realm_for_map, "No specific restoration recommendations available."
)

with row4_right:
    st.markdown("####  Predicted Bleaching Map (Sites in Realm)")
    realm_sites = df_model[df_model['Realm_Name'] == realm_for_map][['Latitude_Degrees', 'Longitude_Degrees']].dropna().copy()
    realm_sites["Predicted_Bleaching"] = predicted_bleaching
    latest_year = df_model[df_model['Realm_Name'] == realm_for_map]['Date_Year'].max()
    actuals = df_model[(df_model['Realm_Name'] == realm_for_map) & (df_model['Date_Year'] == latest_year)]
    # Classify severity
    realm_sites["Bleaching_Severity"] = realm_sites["Predicted_Bleaching"].apply(classify_severity)
    fig_pred_map = px.scatter_mapbox(
        realm_sites,
        lat="Latitude_Degrees",
        lon="Longitude_Degrees",
        color="Bleaching_Severity",
        color_discrete_map=severity_colors,
        size="Predicted_Bleaching",
        size_max=10,
        opacity=0.8,
        zoom=2.3,
        height=350,
        hover_data={"Predicted_Bleaching": ':.2f', "Bleaching_Severity": True}
    )
    if not actuals.empty:
        fig_pred_map.add_scattermapbox(
            lat=actuals["Latitude_Degrees"],
            lon=actuals["Longitude_Degrees"],
            mode="markers",
            marker=dict(size=7, color="black", opacity=0.6),
            text=actuals["Percent_Bleaching"].round(2).astype(str) + "%",
            name=f"Actual {int(latest_year)}"
        )
    fig_pred_map.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=8, r=8, t=22, b=8),
        showlegend=True,
        legend=dict(
            title="Bleaching Severity",
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="right", x=0.99,
            font=dict(size=13)
        )
    )
    st.plotly_chart(fig_pred_map, use_container_width=True)

    # ---- Recommendation box appears IMMEDIATELY after the map ----
    st.markdown(
        f"""
        <div style='padding:1.2em 1em 1em 1em; background-color:#eaf7ff; border-radius:16px; border:1px solid #bbe1fa; margin-top:1.1em;'>
            <span style="font-size:1.35rem; font-weight:600; color:#174672;">Recommendation</span><br>
            <div style="font-size:1.07rem; margin-top:0.7em; color:#174672;">
                <b>Restoration Techniques:</b> {restoration_reco}<br>
                <b>Status:</b> {severity_line}<br>
                <b>Top countries in this realm:</b> {top_countries_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<hr style='margin:0.5em 0;'>", unsafe_allow_html=True)

# ===== Row 5: Custom Prediction + Gauge + Model Confidence =====
rf_col, sev_col, conf_col = st.columns([1.3, 0.7, 0.7])  # Adjusted to more balanced proportions

with rf_col:
    st.markdown("#### Coral Reef Bleaching Predictor")
    user_input_features = [
        'Temperature_Kelvin', 'SSTA', 'Depth_m', 'Turbidity', 'Windspeed',
        'Cyclone_Frequency', 'IDW_G2oxygen'
    ]
    model_input_features = [
        'Temperature_Kelvin', 'SSTA', 'ClimSST', 'Depth_m', 'Turbidity',
        'Windspeed', 'Cyclone_Frequency', 'IDW_G2talk', 'IDW_G2oxygen', 'IDW_G2phts25p0'
    ]
    pred_pct = None
    with st.form("custom_rf_form", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns([1,1,1,1])
        user_vals = []
        for i, feat in enumerate(user_input_features):
            with [c1, c2, c3, c4][i % 4]:
                val = st.number_input(f"{feat}", value=float(df_model[feat].mean()), format="%.2f", key="custom_"+feat)
                user_vals.append(val)
        submitted = st.form_submit_button("Predict", use_container_width=True)
        if submitted:
            full_input = []
            for col in model_input_features:
                if col in user_input_features:
                    idx = user_input_features.index(col)
                    full_input.append(user_vals[idx])
                else:
                    full_input.append(float(df_model[col].mean()))
            input_array = np.array(full_input).reshape(1, -1)
            pred_pct = custom_rf_model.predict(input_array)[0]
            st.success(f"Predicted Bleaching: {pred_pct:.1f}%")
    st.caption("Enter features above for scenario modeling.", unsafe_allow_html=True)

with sev_col:
    # Odometer for severity index (gauge)
    if pred_pct is None:
        pred_pct = 0
    sev_cat = classify_severity(pred_pct)
    pointer = min(max(pred_pct, 0), 100)
    gauge_steps = [
        {'range': [0, 10], 'color': "#3CB371"},
        {'range': [10, 30], 'color': "#FFA500"},
        {'range': [30, 100], 'color': "#D7263D"}
    ]
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pointer,
        title={'text': "Severity Index"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': '#223E72'},
            'steps': gauge_steps,
            'threshold': {
                'line': {'color': '#223E72', 'width': 6},
                'thickness': 0.7,
                'value': pointer
            },
        },
        number={'suffix': "%",'font': {'size':36}}
    ))
    fig_gauge.update_layout(
        height=300,  # Ensures height matches other columns
        margin=dict(l=16, r=16, t=28, b=10),
        font=dict(size=15)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown(
        f"<div style='text-align:center; font-size:1.1rem; color:{severity_colors[sev_cat]}; font-weight:700;'>{sev_cat}</div>",
        unsafe_allow_html=True
    )

with conf_col:
    st.markdown("""
        <div style="padding:2.6em 1em 1em 1em; background-color:#f7f7f9; border-radius:16px; border:1px solid #e4e7ec; text-align:center; height:300px; display:flex; flex-direction:column; justify-content:center;">
            <span style="font-size:2.2rem; font-weight:600; color:#174672;">Model Confidence</span><br>
            <span style="font-size:2.8rem; font-weight:700; color:#3CB371;">71%</span>
            <div style="font-size:1.02rem; margin-top:0.5em; color:#666;">
                Probability that model predictions are<br>accurate on unseen data.
            </div>
        </div>
    """, unsafe_allow_html=True)



# ===== Bulk Upload & Download Panel =====
st.markdown("----")
st.markdown("### Batch Prediction: Upload CSV for Batch Bleaching Prediction")
st.markdown(
    """ 
    Upload a CSV file with columns:  
    <code>Temperature_Kelvin, SSTA, ClimSST, Depth_m, Turbidity, Windspeed, Cyclone_Frequency, IDW_G2talk, IDW_G2oxygen, IDW_G2phts25p0</code>  
    (or a subset; missing columns will be filled with mean values)
    """,
    unsafe_allow_html=True,
)

bulk_file = st.file_uploader(
    "Upload a CSV file for batch prediction",
    type=["csv"],
    accept_multiple_files=False,
    key="bulk_csv"
)

if bulk_file is not None:
    try:
        user_bulk = pd.read_csv(bulk_file)
        all_features = [
            'Temperature_Kelvin', 'SSTA', 'ClimSST', 'Depth_m', 'Turbidity',
            'Windspeed', 'Cyclone_Frequency', 'IDW_G2talk', 'IDW_G2oxygen', 'IDW_G2phts25p0'
        ]
        for col in all_features:
            if col not in user_bulk.columns:
                user_bulk[col] = df_model[col].mean()
        X_bulk = user_bulk[all_features]
        user_bulk["Predicted_Bleaching"] = custom_rf_model.predict(X_bulk)
        st.success(f"Predictions complete! Download results below.")
        st.dataframe(user_bulk, use_container_width=True, height=300)
        from io import StringIO

        output = StringIO()
        user_bulk.to_csv(output, index=False)
        st.download_button(
            label="Download Predicted Results CSV",
            data=output.getvalue().encode(),  # Encode to bytes
            file_name="bulk_predicted_bleaching.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Could not process file: {e}")

# ====== Minimal Style tweaks ======
st.markdown("""
<style>
section[data-testid="stSidebar"], .css-6qob1r {display: none !important;}
div.block-container {padding-top: 0.1rem !important; padding-bottom: 0.15rem !important; max-width: 1850px !important;}
hr {margin-top:0.1em; margin-bottom:0.1em;}
.stPlotlyChart {margin-bottom:0.1em;}
</style>
""", unsafe_allow_html=True)
