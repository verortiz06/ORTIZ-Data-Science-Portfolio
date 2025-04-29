# -----------------------------------------------
# Importing All Necessary Libraries
# ----------------------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

# -----------------------------------------------
# App Information
# ----------------------------------------------
st.title("Unsupervised Machine Learning Playground! 🛝") # Creating a title for the app
st.markdown(""" 
## 📋 About This Application:
This interactive application allows you to upload your own dataset, learn about different 
methods of unsupervised machine learning, experiment with hyperparameters, and observe how you can affect 
the model's training and performance.
""") # App description and explanation
st.info("Let's build a machine learning model!")


# t-sne for visualizations??? reduces dimensions understanding that it will go into a visualization later; but PCA still works pretty well
# add feedback tab
# File information on the sidebar with # rows, # columns, 
# step 1: load and preprocess the data
# adding hovering information on plots with plotly

# -----------------------------------------------
# 📁 Dataset Upload
# -----------------------------------------------
st.sidebar.header("Step 1: Upload or Select Dataset")
sample_datasets = {
    "Iris Dataset": sns.load_dataset("iris").drop("species", axis=1),
    "Palmer's Penguins": sns.load_dataset("penguins").drop(columns=["species", "island", "sex"]).dropna()  # Drop categorical cols and missing values
}

dataset_choice = st.sidebar.selectbox("Dataset Source", ["Upload Your Own"] + list(sample_datasets))
if dataset_choice == "Upload Your Own":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file.")
        st.stop()
else:
    df = sample_datasets[dataset_choice]

st.write("### 🔍 Dataset Preview")
st.dataframe(df.head())

# -----------------------------------------------
# Feature Selection + Scaling
# -----------------------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
selected_features = st.multiselect("Select Features to Use", numeric_cols, default=numeric_cols)
X = df[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------
# Choose Model
# -----------------------------------------------
st.sidebar.header("Step 2: Choose a Model")
model_type = st.sidebar.selectbox("Model Type", ["K-Means Clustering", "Hierarchical Clustering", "PCA"])

# -----------------------------------------------
# Hyperparameters and Model Training
# -----------------------------------------------
if model_type == "K-Means Clustering":
    k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_scaled)
    
    st.subheader("📊 Cluster Scatterplot")
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X_scaled)
    plt.figure()
    scatter = plt.scatter(pca_components[:, 0], pca_components[:, 1], c=labels, cmap="viridis", s=50)
    plt.scatter(pca_components[:, 0], pca_components[:, 1], c=labels, cmap="viridis", s=50)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("K-Means Cluster Visualization")
    plt.legend(*scatter.legend_elements(), title="Clusters")  # Legend for the clusters
    st.pyplot(plt)

    # Silhouette score and plot
    silhouette = silhouette_score(X_scaled, labels)
    st.metric("Silhouette Score", f"{silhouette:.3f}")
    st.subheader("📉 Silhouette Score Plot")
    # ADD THIS HERE!!!!!!!


    # Elbow plot
    st.subheader("📉 Elbow ")
    sse = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42).fit(X_scaled)
        sse.append(kmeans.inertia_)
    plt.figure()
    plt.plot(range(1, 11), sse, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
    plt.title("Elbow Plot for Optimal k")
    plt.grid(True)
    st.pyplot(plt)

elif model_type == "Hierarchical Clustering":
    method = st.sidebar.selectbox("Method", ["ward", "complete", "average", "single"])
    model = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels = model.fit_predict(X_scaled)

    st.subheader("📊 Cluster Scatterplot (via PCA)")
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled)
    plt.figure()
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='plasma')
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='plasma')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title(f"Hierarchical Clustering (Method: {method})")
    plt.grid(True)
    st.pyplot(plt)

    # Dendrogram
    st.subheader("🌿 Dendrogram")
    Z = linkage(X_scaled, method=method)
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    st.pyplot(plt)

elif model_type == "PCA":
    n_components = st.sidebar.slider("Number of Components", 1, min(10, X.shape[1]), 2)
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_
    st.subheader("📈 Explained Variance by Component")
    plt.figure()
    plt.plot(range(1, n_components+1), explained_var, marker='o')
    plt.title("Explained Variance")
    plt.xlabel("Component")
    plt.ylabel("Variance Ratio")
    st.pyplot(plt)

    st.subheader("🧭 PCA Scatterplot")
    if n_components >= 2:
        plt.figure()
        plt.scatter(components[:, 0], components[:, 1], alpha=0.6)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        st.pyplot(plt)

# -----------------------------------------------
# Cluster Labels
# -----------------------------------------------
if model_type in ["K-Means Clustering", "Hierarchical Clustering"]:
    st.write("### 📌 Cluster Assignments")
    cluster_df = df.copy()
    cluster_df["Cluster"] = labels
    st.dataframe(cluster_df)