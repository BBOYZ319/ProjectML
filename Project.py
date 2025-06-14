import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import re

st.set_page_config(layout="wide", page_title="Scatter Plot Kluster Film Netflix")
st.title("Visualisasi Scatter Plot Kluster Film Netflix")

@st.cache_data(show_spinner="Memuat dan memproses dataset...")
def load_and_preprocess_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Gagal membaca dataset: {e}")
        return None, None, None, None

    def parse_duration_to_minutes(duration_str, show_type):
        if pd.isna(duration_str):
            return np.nan
        duration_str = str(duration_str).strip()
        if show_type == 'Movie':
            match = re.match(r'(\d+)\s*min', duration_str)
            if match:
                return float(match.group(1))
        return np.nan

    df['duration_numeric'] = df.apply(lambda row: parse_duration_to_minutes(row['duration'], row['type']), axis=1)
    df['rating_encoded'], _ = pd.factorize(df['rating'])

    required_cols = ['rating_encoded', 'duration_numeric', 'release_year']
    df_cleaned = df[df['type'] == 'Movie'].dropna(subset=required_cols)

    if len(df_cleaned) == 0:
        return None, None, None, None

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_cleaned[required_cols]), columns=required_cols, index=df_cleaned.index)

    return df_cleaned, df_scaled, scaler, required_cols

@st.cache_resource(show_spinner="Melakukan K-Means Clustering...")
def perform_kmeans(df_scaled, features_for_clustering, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_scaled['cluster'] = kmeans.fit_predict(df_scaled[features_for_clustering])
    return df_scaled, kmeans

# Ganti bagian ini: baca langsung dari file lokal
csv_path = "netflix_titles.csv"
df_original, df_scaled, scaler_model, used_features = load_and_preprocess_data(csv_path)

if df_original is not None:
    selected_n_clusters = st.sidebar.slider("Pilih Jumlah Kluster (K)", min_value=2, max_value=10, value=3)

    clustered_df_scaled, kmeans_model = perform_kmeans(df_scaled.copy(), used_features, n_clusters=selected_n_clusters)
    df_original['cluster'] = clustered_df_scaled['cluster']

    st.subheader("Scatter Plot Hasil Klustering")

    plot_features = []
    if len(used_features) >= 3:
        plot_features = [
            (used_features[0], used_features[1]),
            (used_features[1], used_features[2]),
            (used_features[0], used_features[2])
        ]
    elif len(used_features) == 2:
        plot_features = [(used_features[0], used_features[1])]

    if plot_features:
        fig, axes = plt.subplots(1, len(plot_features), figsize=(8 * len(plot_features), 7))
        if len(plot_features) == 1:
            axes = [axes]
        plt.style.use('seaborn-v0_8-darkgrid')

        for i, (feat_x, feat_y) in enumerate(plot_features):
            sns.scatterplot(data=df_original, x=feat_x, y=feat_y, hue='cluster', palette='viridis', ax=axes[i], s=50, alpha=0.7)
            axes[i].set_title(f'{feat_x.replace("_", " ").title()} vs. {feat_y.replace("_", " ").title()}')
            axes[i].set_xlabel(feat_x.replace("_", " ").title())
            axes[i].set_ylabel(feat_y.replace("_", " ").title())
            axes[i].legend(title='Cluster')
        st.pyplot(fig)
    else:
        st.warning("Tidak cukup fitur untuk visualisasi scatter plot.")
else:
    st.warning("Data tidak valid atau kosong setelah diproses.")
