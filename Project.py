import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import davies_bouldin_score
import os
import re


@st.cache_data(show_spinner="Memuat dan memproses dataset...")
def load_and_preprocess_data():
    """
    Memuat dataset dari file CSV lokal, membersihkan data yang hilang,
    dan melakukan normalisasi sesuai laporan.
    """
    dataset_file_path = 'netflix_titles.csv'

    if not os.path.exists(dataset_file_path):
        st.error(f"File dataset '{dataset_file_path}' tidak ditemukan. Pastikan file CSV sudah didownload dan diletakkan di direktori yang sama dengan aplikasi Streamlit Anda.")
        st.stop()
        return None, None, None, None

    try:
        df = pd.read_csv(dataset_file_path)
    except Exception as e:
        st.error(f"Gagal membaca dataset dari '{dataset_file_path}': {e}")
        st.stop()
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

    required_cols_for_this_csv = ['rating_encoded', 'duration_numeric', 'release_year']
    df_cleaned = df[df['type'] == 'Movie'].dropna(subset=required_cols_for_this_csv)

    if len(df_cleaned) == 0:
        st.warning("Tidak ada data yang tersisa setelah pembersihan. Pastikan dataset memiliki data 'Movie' yang valid dan kolom yang diharapkan.")
        return None, None, None, None

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_cleaned[required_cols_for_this_csv]), columns=required_cols_for_this_csv, index=df_cleaned.index)

    return df_cleaned, df_scaled, scaler, required_cols_for_this_csv


@st.cache_resource(show_spinner="Melakukan K-Means Clustering...")
def perform_kmeans(df_scaled, features_for_clustering, n_clusters=3):
    """
    Melakukan K-Means clustering pada data yang dinormalisasi.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_scaled['cluster'] = kmeans.fit_predict(df_scaled[features_for_clustering])
    return df_scaled, kmeans


def calculate_dbi(X, labels):
    """
    Menghitung Davies-Bouldin Index.
    Semakin kecil atau mendekati nol nilai DBI, semakin baik.
    """
    if len(np.unique(labels)) < 2:
        return np.inf
    return davies_bouldin_score(X, labels)


st.set_page_config(layout="wide", page_title="Analisis Kluster Film Netflix Populer")

st.title("Analisis Kluster Film Netflix Populer dengan K-Means")

st.markdown("""
Aplikasi ini melakukan analisis klustering pada film-film populer Netflix menggunakan algoritma K-Means.
Tujuannya adalah untuk dapat mengidentifikasi film-film popular pada Netflix menjadi beberapa kelompok.

**Catatan Penting:** Dataset `netflix_titles.csv` yang diunggah tidak memiliki kolom 'votes' atau 'rating' (numerik) seperti yang dijelaskan dalam laporan.
Untuk tujuan demonstrasi dan agar kode dapat dijalankan, kami menggunakan `rating` (content rating yang di-*encode*), `duration` (di-*parse* ke menit), dan `release_year` sebagai pengganti 'votes'.
Oleh karena itu, hasil klustering dan interpretasinya akan **berbeda** dari yang dijelaskan dalam laporan asli.
""")

df_original, df_scaled, scaler_model, used_features = load_and_preprocess_data()

if df_original is not None and df_scaled is not None and used_features is not None:
    st.header("Analisis Klustering")

    selected_n_clusters = st.sidebar.slider("Pilih Jumlah Kluster (K)", min_value=2, max_value=10, value=3)

    clustered_df_scaled, kmeans_model = perform_kmeans(df_scaled.copy(), used_features, n_clusters=selected_n_clusters)
    df_original['cluster'] = clustered_df_scaled['cluster']

    st.subheader("Visualisasi Hasil Klustering (Scatter Plot)")
    st.write("Berikut tampilan grafik scatter plot hasil pemodelan atribut yang digunakan:")

    plot_features = []
    if len(used_features) >= 3:
        plot_features = [
            (used_features[0], used_features[1]),
            (used_features[1], used_features[2]),
            (used_features[0], used_features[2])
        ]
    elif len(used_features) == 2:
        plot_features = [(used_features[0], used_features[1])]
    else:
        st.warning("Tidak cukup fitur untuk membuat scatter plot.")

    if plot_features:
        fig, axes = plt.subplots(1, len(plot_features), figsize=(8 * len(plot_features), 7))
        if len(plot_features) == 1:
            axes = [axes]
        plt.style.use('seaborn-v0_8-darkgrid')

        for i, (feat_x, feat_y) in enumerate(plot_features):
            sns.scatterplot(data=df_original, x=feat_x, y=feat_y, hue='cluster', palette='viridis', ax=axes[i], s=50, alpha=0.7)
            axes[i].set_title(f'{feat_x.replace("_", " ").title()} vs. {feat_y.replace("_", " ").title()} per Cluster')
            axes[i].set_xlabel(feat_x.replace("_", " ").title())
            axes[i].set_ylabel(feat_y.replace("_", " ").title())
            axes[i].legend(title='Cluster')
        st.pyplot(fig)
    else:
        st.write("Tidak ada visualisasi yang dapat dibuat karena kurang dari 2 fitur yang tersedia.")


    st.header("Evaluasi dan Interpretasi Kluster")

    st.subheader("Davies-Bouldin Index (DBI)")
    st.write("Berikut adalah nilai Davies-Bouldin Index (DBI) untuk hasil klustering:")
    dbi_score = calculate_dbi(df_scaled[used_features], clustered_df_scaled['cluster'])
    st.write(f"* Nilai DBI untuk {selected_n_clusters} Kluster: **{dbi_score:.3f}**")
    st.write("  * Mengacu pada prinsip DBI, nilai yang dikategorikan sebagai nilai baik adalah semakin kecil atau mendekati nol.")
    st.write("  * Nilai DBI yang dihitung di aplikasi ini (untuk atribut yang berbeda) mungkin akan berbeda dari nilai DBI dari laporan asli yang disebutkan dalam tabel berikut:")
    st.markdown("""
    | Jumlah Kluster | Nilai DBI (dari laporan) |
    |----------------|--------------------------|
    | 3              | -0.441                   |
    | 4              | -0.494                   |
    | 5              | -0.443                   |
    | 6              | -0.447                   |
    | 7              | -0.450                   |
    """)
    st.write("  * Dari data Tabel 1 dalam laporan, terdapat 5 kategori kluster yang memiliki nilai DBI, dan jumlah cluster dengan nilai DBI yang paling baik adalah cluster 3, dengan nilai DBI yang mendekati nol yaitu -0.441. Oleh karena itu, 3 kluster digunakan untuk pengelompokkan.")


    st.subheader("Jumlah Anggota pada Setiap Kluster")
    st.write("Distribusi jumlah anggota pada setiap kluster adalah sebagai berikut:")
    cluster_counts = df_original['cluster'].value_counts().sort_index()
    st.dataframe(cluster_counts.rename("Jumlah Item").to_frame().T)
    st.write(f"* Total jumlah item: {len(df_original)}.")
    st.write("* Pada Gambar 8 dalam laporan, salah satu cluster memiliki jumlah data yang lebih besar dibandingkan dengan cluster lainnya.")

    st.subheader("Rata-rata Centroid Kluster untuk Setiap Atribut")
    st.write("Berikut adalah rata-rata centroid kluster untuk setiap atribut yang digunakan dalam analisis:")
    cluster_means = df_original.groupby('cluster')[used_features].mean()
    display_cols = {col: col.replace("_", " ").title() for col in used_features}
    cluster_means = cluster_means.rename(columns=display_cols)
    st.dataframe(cluster_means)

    st.subheader("Interpretasi Kluster")
    st.write("Dari data dan tabel di atas, setiap kluster dari atribut yang digunakan dapat dibedakan sebagai berikut:")

    for idx, row in cluster_means.iterrows():
        st.markdown(f"**Kluster {idx}:**")
        for col_raw in used_features:
            col_display = col_raw.replace("_", " ").title()
            st.markdown(f"* {col_display}: {row[col_display]:.3f}")