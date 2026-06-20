from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Car Price ML Dashboard", page_icon="🚗", layout="wide")
BASE = Path(__file__).resolve().parent

# =========================
# Loading (Backend & Caching)
# =========================
def path_of(*names):
    for n in names:
        for p in [BASE / n, BASE / "models" / n, BASE / "car" / n]:
            if p.exists():
                return p
    return None

@st.cache_resource
def load_pkl(name):
    p = path_of(name)
    return (joblib.load(p), p) if p else (None, None)

@st.cache_data
def load_data():
    files = [
        ("cleaned_cars_data.csv", pd.read_csv),
        ("cleaned_cars_data.xlsx", pd.read_excel),
        ("cleaned_cars_data.pkl", pd.read_pickle),
    ]
    for name, reader in files:
        p = path_of(name)
        if p:
            return reader(p), p
    return None, None

# Load all models including the KNN Recommendation Engine
xgb, xgb_p = load_pkl("xgb_model.pkl")
lgbm, lgbm_p = load_pkl("lgbm_model.pkl")
scaler, scaler_p = load_pkl("scaler_supervised.pkl")
df_encoded, encoded_p = load_pkl("df_encoded.pkl")
scaler_unsup, scaler_unsup_p = load_pkl("scaler_unsupervised.pkl")
pca, pca_p = load_pkl("pca_clustering.pkl")
kmeans, kmeans_p = load_pkl("kmeans_model.pkl")
gmm, gmm_p = load_pkl("gmm_model.pkl")
knn, knn_p = load_pkl("knn_engine.pkl") 
df, data_p = load_data()

# =========================
# Helpers
# =========================
def original_price(s):
    s = pd.to_numeric(s, errors="coerce")
    return np.expm1(s) if not s.dropna().empty and s.max() < 20 else s

def money(x):
    return f"£{x:,.0f}"

def uniq(col, fallback):
    if df is not None and col in df.columns:
        vals = sorted(df[col].dropna().astype(str).unique())
        return list(vals) if len(vals) else fallback
    return fallback

def features():
    if df_encoded is None:
        return []
    cols = list(df_encoded.drop(columns=["price"]).columns) if "price" in df_encoded.columns else list(df_encoded.columns)
    return list(scaler.feature_names_in_) if scaler is not None and hasattr(scaler, "feature_names_in_") else cols

FEATURES = features()

def make_input(values):
    row = pd.DataFrame(0.0, index=[0], columns=FEATURES)

    age = 2026 - int(values["year"])
    numeric = {
        "year": values["year"],
        "mileage": values["mileage"],
        "tax": values["tax"],
        "mpg": values["mpg"],
        "engineSize": values["engine_size"],
        "age": age,
        "mileage_per_year": values["mileage"] if age <= 0 else values["mileage"] / age,
        "engine_efficiency": values["mpg"] if values["engine_size"] <= 0 else values["mpg"] / values["engine_size"],
    }
    for c, v in numeric.items():
        if c in row.columns:
            row.loc[0, c] = v

    cats = {
        "brand": values["brand"],
        "model": values["model"],
        "fuelType": values["fuel"],
        "transmission": values["transmission"],
    }
    for c, v in cats.items():
        dummy = f"{c}_{v}"
        if dummy in row.columns:
            row.loc[0, dummy] = 1

    return row

def predict(row, model):
    scaled = pd.DataFrame(scaler.transform(row), columns=row.columns)
    return max(float(np.expm1(model.predict(scaled)[0])), 0)

# =========================
# PCA & FAST Segmentation Helpers (OPTIMIZED)
# =========================
@st.cache_data
def pca_data():
    if not all([df_encoded is not None, scaler_unsup, pca, kmeans]):
        return None, None, None
    try:
        cols = list(scaler_unsup.feature_names_in_) if hasattr(scaler_unsup, "feature_names_in_") else [
            c for c in df_encoded.columns if c not in ["price", "year", "mileage"]
        ]
        x = df_encoded.reindex(columns=cols, fill_value=0)
        z = pca.transform(scaler_unsup.transform(x))
        labels = kmeans.predict(z)
        return z, labels, getattr(kmeans, "cluster_centers_", None)
    except Exception:
        return None, None, None

@st.cache_data
def segment_table():
    if df_encoded is None or df is None:
        return None
    z, labels, _ = pca_data()
    if labels is None or len(labels) != len(df_encoded):
        return None
    out = pd.DataFrame({"Segment": labels})
    if "price" in df.columns and len(df) == len(out):
        out["Price"] = original_price(df["price"]).values
    elif "price" in df_encoded.columns:
        out["Price"] = original_price(df_encoded["price"]).values
    else:
        out["Price"] = np.nan
    summary = out.groupby("Segment").agg(
        Cars_Count=("Segment", "size"),
        Average_Price=("Price", "mean")
    ).reset_index()
    summary = summary.sort_values("Average_Price").reset_index(drop=True)
    names = ["Budget / Economy", "Mid-Range", "Premium / High-Value"]
    if len(summary) > 3:
        names = ["Budget / Economy"] + ["Mid-Range"] * (len(summary) - 2) + ["Premium / High-Value"]
    summary["Suggested Meaning"] = names[:len(summary)]
    return summary

@st.cache_data
def get_cluster_mapping():
    tbl = segment_table()
    if tbl is None:
        return {}
    return dict(zip(tbl["Segment"], tbl["Suggested Meaning"]))

def segment_name(cluster):
    mapping = get_cluster_mapping()
    return mapping.get(cluster, f"Segment {cluster}")

def segment_and_recommend(row):
    if not all([scaler_unsup, pca, kmeans]):
        st.info("Market segmentation files are missing.")
        return None
    try:
        cols = list(scaler_unsup.feature_names_in_) if hasattr(scaler_unsup, "feature_names_in_") else [
            c for c in row.columns if c not in ["year", "mileage", "price"]
        ]
        x = row.reindex(columns=cols, fill_value=0)
        scaled_unsup = scaler_unsup.transform(x)
        
        z = pca.transform(scaled_unsup)
        cluster = int(kmeans.predict(z)[0])
        
        st.metric("Predicted Market Category", segment_name(cluster))
        
        if gmm is not None:
            st.caption("Soft Clustering Probabilities (GMM):")
            probs = pd.DataFrame({
                "Segment": [segment_name(i) for i in range(len(gmm.predict_proba(z)[0]))],
                "Probability": gmm.predict_proba(z)[0]
            }).sort_values("Probability", ascending=False)
            
            for i, r in probs.iterrows():
                st.progress(r["Probability"], text=f"{r['Segment']}: {r['Probability']:.1%}")

        if knn is not None and df is not None:
            st.markdown("---")
            st.subheader("💡 Top 5 Recommended Alternative Cars")
            
            distances, indices = knn.kneighbors(scaled_unsup, n_neighbors=5)
            recommended_cars = df.iloc[indices[0]].copy()
            if "price" in recommended_cars.columns:
                 recommended_cars["price"] = original_price(recommended_cars["price"])
            
            cols_to_show = ['brand', 'model', 'year', 'price', 'mileage', 'transmission', 'fuelType']
            valid_cols = [c for c in cols_to_show if c in recommended_cars.columns]
            
            st.dataframe(
                recommended_cars[valid_cols].style.format({"price": "£{:.0f}", "mileage": "{:,.0f}"}),
                use_container_width=True,
                hide_index=True
            )
            
    except Exception as e:
        st.warning(f"Segmentation/Recommendation error: {e}")

def prediction_form():
    brands = uniq("brand", ["Audi", "BMW", "Ford", "Hyundai", "Mercedes", "Skoda", "Toyota", "Vauxhall", "Volkswagen"])
    fuels = uniq("fuelType", ["Diesel", "Petrol", "Hybrid", "Electric", "Other"])
    transmissions = uniq("transmission", ["Manual", "Automatic", "Semi-Auto", "Other"])

    c1, c2, c3, c4 = st.columns(4)
    brand = c1.selectbox("Brand", brands)

    if df is not None and {"brand", "model"}.issubset(df.columns):
        models = sorted(df.loc[df["brand"].astype(str) == brand, "model"].dropna().astype(str).unique())
    else:
        models = sorted(c.replace("model_", "") for c in FEATURES if str(c).startswith("model_"))
    model = c2.selectbox("Model", models or ["Other"])
    fuel = c3.selectbox("Fuel Type", fuels)
    transmission = c4.selectbox("Transmission", transmissions)

    c5, c6, c7, c8, c9 = st.columns(5)
    vals = {
        "brand": brand,
        "model": model,
        "fuel": fuel,
        "transmission": transmission,
        "year": c5.number_input("Year", 1990, 2026, 2019),
        "mileage": c6.number_input("Mileage", 0, 500000, 30000, step=1000),
        "tax": c7.number_input("Tax (£)", 0, 1000, 145, step=5),
        "mpg": c8.number_input("MPG", 1.0, 250.0, 50.0, step=0.5),
        "engine_size": c9.number_input("Engine Size", 0.0, 10.0, 1.6, step=0.1),
    }

    models_map = {}
    if xgb is not None:
        models_map["XGBoost"] = xgb
    if lgbm is not None:
        models_map["LightGBM"] = lgbm

    vals["model_name"] = st.radio("Prediction Model", list(models_map), horizontal=True)
    vals["model_obj"] = models_map[vals["model_name"]]
    return vals

# =========================
# Interactive Plotly Visualizations (PROTECTED & MAPPED)
# =========================
def plot_importance_plotly():
    models = [(n, m) for n, m in [("XGBoost", xgb), ("LightGBM", lgbm)] if m is not None and hasattr(m, "feature_importances_")]
    if not models:
        st.info("Feature importance data is currently unavailable.")
        return

    cols = st.columns(len(models))
    for idx, (name, model) in enumerate(models):
        with cols[idx]:
            imp = pd.Series(model.feature_importances_, index=FEATURES).nlargest(10).reset_index()
            imp.columns = ["Feature", "Importance"]
            imp = imp.sort_values(by="Importance", ascending=True)
            
            fig = px.bar(
                imp, x="Importance", y="Feature", orientation="h",
                title=f"{name}: Top 10 Feature Importance",
                color="Importance", color_continuous_scale="Viridis"
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False, height=450, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

def plot_segments_2d_plotly():
    z, labels, centers = pca_data()
    if z is None or z.shape[1] < 2:
        st.info("2D segmentation plot is unavailable.")
        return

    n = min(5000, len(z))
    idx = np.linspace(0, len(z) - 1, n).astype(int)
    
    plot_df = pd.DataFrame({
        'PC1': z[idx, 0],
        'PC2': z[idx, 1],
        'Cluster': [segment_name(lbl) for lbl in labels[idx]]
    })

    fig = px.scatter(
        plot_df, x='PC1', y='PC2', color='Cluster',
        title=f"PCA-Optimized Segmentation (K={len(np.unique(labels))})",
        opacity=0.7, color_discrete_sequence=px.colors.qualitative.Prism
    )
    
    if centers is not None and centers.shape[1] >= 2:
        fig.add_trace(go.Scatter(
            x=centers[:, 0], y=centers[:, 1], mode='markers',
            marker=dict(size=15, color='red', symbol='x', line=dict(width=2, color='DarkSlateGrey')),
            name='Centroids'
        ))
        
    st.plotly_chart(fig, use_container_width=True)

def plot_segments_3d_plotly():
    z, labels, centers = pca_data()
    if z is None or z.shape[1] < 3:
        st.info("3D segmentation plot is unavailable.")
        return

    n = min(5000, len(z))
    idx = np.linspace(0, len(z) - 1, n).astype(int)
    
    plot_df = pd.DataFrame({
        'PC1': z[idx, 0],
        'PC2': z[idx, 1],
        'PC3': z[idx, 2],
        'Cluster': [segment_name(lbl) for lbl in labels[idx]]
    })

    fig = px.scatter_3d(
        plot_df, x='PC1', y='PC2', z='PC3', color='Cluster',
        title="Market Segmentation in 3D Space",
        opacity=0.6, color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig.update_traces(marker=dict(size=3))
    
    if centers is not None and centers.shape[1] >= 3:
        fig.add_trace(go.Scatter3d(
            x=centers[:, 0], y=centers[:, 1], z=centers[:, 2], mode='markers',
            marker=dict(size=8, color='red', symbol='cross'),
            name='Centroids'
        ))
        
    fig.update_layout(scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))
    st.plotly_chart(fig, use_container_width=True)

# =========================
# UI Header (Rendered ONCE globally)
# =========================
st.title("🚗 Car Price ML & Analytics Dashboard")
st.caption("End-to-End Pipeline: EDA -> Feature Engineering -> Supervised Pricing -> Unsupervised Market Segmentation")

# Main architecture tabs (Clean Layout Navigation)
tab1, tab2, tab3 = st.tabs([
    "💰 Smart Pricing & Recommendations", 
    "🤖 Clustering Analytics", 
    "📈 Model Benchmarking"
])

with tab1:
    st.subheader("Smart Pricing & Market Recommendations")
    vals = prediction_form()

    if st.button("Predict Price", type="primary", use_container_width=True):
        row = make_input(vals)
        price = predict(row, vals["model_obj"])
        age = 2026 - int(vals["year"])
        mileage_year = vals["mileage"] if age <= 0 else vals["mileage"] / age

        st.success(f"Estimated price for {vals['brand']} {vals['model']}: {money(price)}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Price", money(price))
        c2.metric("Selected Model", vals["model_name"])
        c3.metric("Car Age", f"{age} years")
        c4.metric("Mileage / Year", f"{mileage_year:,.0f}")

        st.markdown("---")
        segment_and_recommend(row)

with tab2:
    st.subheader("Clustering Analytics (Unsupervised Learning)")
    st.info("PCA was utilized to compress features into 3 principal components. K-Means and GMM classify cars based strictly on core engineering specifications, actively ignoring depreciation variables (Year, Mileage) for pure market segmentation.")
    
    st.subheader("PCA Segmentation — 2D View")
    plot_segments_2d_plotly()
    
    st.markdown("---")
    st.subheader("PCA Segmentation — 3D View (Interactive)")
    plot_segments_3d_plotly()

with tab3:
    st.subheader("Model Benchmarking & Evaluation")
    
    st.subheader("1. Supervised Price Prediction Models")
    benchmark_data = pd.DataFrame({
        "Model": ["XGBoost Regressor", "LightGBM Regressor"],
        "R² Score (%)": ["95.67%", "94.47%"],
        "RMSE (£)": ["2,055.84", "2,322.61"],
        "MAE (£)": ["1,192.41", "1,343.74"],
        "Train Time (s)": ["3.03", "1.36"],
        "Inference Time (s)": ["0.0385", "0.0706"]
    })
    st.dataframe(benchmark_data, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("2. Feature Importance")
    plot_importance_plotly()
        
    st.markdown("---")
    st.subheader("3. Unsupervised Clustering Metrics")
    clustering_metrics = pd.DataFrame({
        "Algorithm": ["PCA + K-Means", "PCA + GMM"],
        "Optimal K / Components": ["K=4, PCA=3", "K=4, Covariance=Full"],
        "Silhouette Score": ["0.4429", "N/A (Soft Clustering)"],
        "Davies-Bouldin Index": ["0.8144", "N/A"],
        "BIC Score": ["N/A", "311,924.85"]
    })
    st.dataframe(clustering_metrics, use_container_width=True, hide_index=True)