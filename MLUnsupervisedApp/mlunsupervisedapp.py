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


# add feedback tab
# File information on the sidebar with # rows, # columns, 
# step 1: load and preprocess the data
# adding hovering information on plots with plotly

# -----------------------------------------------
# Step 1: Choose Dataset
# -----------------------------------------------
st.sidebar.header("Step 1: Upload or Select Dataset")
sample_datasets = {
    "Iris Dataset": sns.load_dataset("iris").drop("species", axis = 1),
    "Palmer's Penguins": sns.load_dataset("penguins").drop(columns = ["island", "sex"]).dropna() # Drop categorical cols and missing values
}

dataset_choice = st.sidebar.selectbox("Dataset Source", ["Upload Your Own"] + list(sample_datasets))
if dataset_choice == "Upload Your Own":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type = ["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file.")
        st.stop()
else:
    df = sample_datasets[dataset_choice]

st.divider()

# -----------------------------------------------
# Dataset Preview
# -----------------------------------------------
st.write("### 🔍 Dataset Preview")
st.markdown("Here you can see the first five rows of the dataset!")
st.dataframe(df.head())

st.divider()

# -----------------------------------------------
# Step 2: Feature Selection + Scaling
# -----------------------------------------------
st.sidebar.header("Step 2: Choose Features")
numeric_cols = df.select_dtypes(include = [np.number]).columns.tolist()
selected_features = st.sidebar.multiselect("Select Features to Use", numeric_cols, default = numeric_cols)

if len(selected_features) < 2:
    st.warning("Please select at least two features to proceed.")
    st.stop()

X = df[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------
# Step 3: Choose Model
# -----------------------------------------------
st.sidebar.header("Step 3: Choose a Model")
model_choice = st.sidebar.selectbox("Model Type", ["K-Means Clustering", "Hierarchical Clustering", "PCA"])
if model_choice == "K-Means Clustering":
    st.sidebar.markdown("""
    **K-Means Clustering** groups data into *k* clusters based on feature similarity. 
                        
    It iteratively assigns a point to the nearest cluster "centroid", which is initially randomly placed, 
    and then updates the positions of the centroids based on the mean of the clusters created.
    """)
if model_choice == "Hierarchical Clustering":
    st.sidebar.markdown("""
    **Hierarchical Clustering** creates a dendogram, which is a sort of hierarchical tree
    that demonstrates how the data are related a different levels.
                        
    We will be using the *ward* linkage method, which means that clusters will be merged in a way that 
    results in the *smallest* increase of within-cluster variance.
                        """)
if model_choice == "PCA":
    st.sidebar.markdown("""
    **PCA (or Principal Component Analysis)** is a method in which you can reduce dimensionality to 2 dimensions. 
    
    This works through the machine finding linear combinations within the features that capture the maximum amount
    of variance. 
                        
    These linear combinations are axes, or *principal components*. We reduce down to 2 components for visualization 
    purposes, which can illustrate for us the influence of the original features.
                        """)


# -----------------------------------------------
# Hyperparameters, Model Training, and Visualizations
# -----------------------------------------------
if model_choice == "K-Means Clustering":
    
    # Step 4: Choosing Hyperparameters
    st.sidebar.header("Step 4: Choose Hyperparameters")
    k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
    
    # Training Model
    model = KMeans(n_clusters = k, random_state = 20)
    labels = model.fit_predict(X_scaled)
    
    # Cluster scatterplot
    st.markdown("## Model Visualizations:")
    st.subheader("📊 Cluster Scatterplot")
    pca = PCA(n_components = 2)
    pca_components = pca.fit_transform(X_scaled)
    plt.figure()
    scatter = plt.scatter(
        pca_components[:, 0], 
        pca_components[:, 1], 
        c = labels, 
        cmap = "Accent", 
        s = 50)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("K-Means Cluster Visualization")
    plt.grid(True)
    plt.legend(*scatter.legend_elements(), title = "Clusters")  # Legend for the clusters
    st.pyplot(plt)

    # True labels comparison
    if "species" in df.columns:
        st.subheader("🎯 Comparing Clusters with True Labels")
        true_labels = df["species"]
        target_names = true_labels.unique()
        label_map = {name: idx for idx, name in enumerate(target_names)}
        y_true = true_labels.map(label_map)

        plt.figure(figsize = (8, 6))
        for i, target_name in enumerate(target_names):
            plt.scatter(
                pca_components[y_true == i, 0],
                pca_components[y_true == i, 1],
                alpha = 0.7,
                edgecolor = 'k',
                label = target_name,
                s = 50
            )
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("True Labels: 2D PCA Projection")
        plt.legend(loc = "best")
        plt.grid(True)
        st.pyplot(plt)

    st.divider()
    st.markdown("## Model Performance Metrics:")


    # -----------------------------------------------
    # Performance Metrics
    # -----------------------------------------------
    # Model Accuracy how to do this?????
    

    st.divider()

    # -----------------------------------------------
    # Evaluating Optimal Number of Clusters
    # -----------------------------------------------
    # Elbow Method (WCSS) and Silhouette Score Plot
    st.markdown("## Evaluating Optimal Number of Clusters:")
    st.subheader("📉 Elbow Method and Silhouette Score")
    sse = []
    sil_scores = []
    k_range = range(2, 11)  # silhouette scores are only valid for k >= 2

    for k_val in k_range:
        kmeans = KMeans(n_clusters = k_val, random_state = 20)
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)
        score = silhouette_score(X_scaled, kmeans.labels_)
        sil_scores.append(score)
    
    best_k = k_range[np.argmax(sil_scores)]
    best_score = max(sil_scores)

    # Elbow Plot
    plt.figure()
    plt.plot(list(k_range), sse, marker = 'o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
    plt.title("Elbow Plot (WCSS) for Optimal k")
    st.pyplot(plt)

    # Silhouette Score
    silhouette = silhouette_score(X_scaled, labels)
    st.metric("Silhouette Score", f"{silhouette:.3f}")

    # Silhouette Plot
    plt.figure()
    plt.plot(list(k_range), sil_scores, marker = 'o', color = 'orange')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score for Optimal k")
    st.pyplot(plt)

    # Optimal k
    st.info(f"Best number of clusters by silhouette score: **{best_k}** (score = {best_score:.3f})")

elif model_choice == "Hierarchical Clustering":
    
    # Step 4: Choose Hyperparameters
    st.sidebar.header("Step 4: Choose Hyperparameters:")
    n_clusters = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
    
    # Training Model
    method = "ward"
    model = AgglomerativeClustering(n_clusters = n_clusters, linkage = "ward")
    labels = model.fit_predict(X_scaled)

    # Cluster scatterplot
    st.markdown("## Model Visualizations:")
    st.subheader("📊 Cluster Scatterplot (via PCA)")
    pca = PCA(n_components = 2)
    reduced = pca.fit_transform(X_scaled)
    plt.figure()
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c = labels, cmap = 'plasma')
    plt.legend(*scatter.legend_elements(), title = "Clusters")
    plt.title("Hierarchical Clustering (Method: ward)")
    plt.grid(True)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    st.pyplot(plt)

    # True labels comparison
    if "species" in df.columns:
        st.subheader("🎯 Comparing Clusters with True Labels")
        true_labels = df["species"]
        target_names = true_labels.unique()
        label_map = {name: idx for idx, name in enumerate(target_names)}
        y_true = true_labels.map(label_map)

        plt.figure(figsize = (8, 6))
        for i, target_name in enumerate(target_names):
            plt.scatter(
                reduced[y_true == i, 0],
                reduced[y_true == i, 1],
                alpha = 0.7,
                edgecolor = 'k',
                label = target_name,
                s = 50
            )
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("True Labels: 2D PCA Projection")
        plt.legend(loc = "best")
        plt.grid(True)
        st.pyplot(plt)

    # Dendrogram
    st.subheader("🌳 Dendrogram")
    Z = linkage(X_scaled, method = method)
    plt.figure(figsize = (10, 5))
    dendrogram(Z)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("?????") # and how on earth do i make this look better
    plt.ylabel("Distance") 
    st.pyplot(plt)
    # what should the x and y labels be and how do i do that????

    st.divider()

    # -----------------------------------------------
    # Evaluating Optimal Number of Clusters
    # -----------------------------------------------
    # Silhouette Elbow
    st.markdown("## Evaluating Optimal Number of Clusters:")

    k_range = range(2, 11)
    sil_scores = []

    for k in k_range:
        hc = AgglomerativeClustering(n_clusters = k, linkage = 'ward')
        labels_k = hc.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels_k)
        sil_scores.append(score)

    best_k = k_range[np.argmax(sil_scores)]

    # Silhouette Score
    silhouette = silhouette_score(X_scaled, labels)
    st.metric("Silhouette Score", f"{silhouette:.3f}")

    # Silhouette Score Plot
    plt.figure(figsize = (7, 4))
    plt.plot(list(k_range), sil_scores, marker = "o", color = 'teal')
    plt.xticks(list(k_range))
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Average Silhouette Score")
    plt.title("Silhouette Analysis for Agglomerative (Ward) Clustering")
    plt.grid(True, alpha = 0.3)
    st.pyplot(plt)

    # Optimal k
    st.info(f"Best number of clusters by silhouette score: **{best_k}** (score = {max(sil_scores):.3f})")

elif model_choice == "PCA":
    st.markdown("## Model Visualizations:")

    # Initalize PCA
    pca = PCA(n_components = 2)
    components = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    # Determining/Preprocessing Data
    if dataset_choice == "Iris Dataset": # If user chooses Iris Dataset
        full_df = sns.load_dataset("iris")
        y = pd.factorize(full_df["species"])[0]
        labels = full_df["species"]
        target_names = labels.unique()
    elif dataset_choice == "Palmer's Penguins": # If user chooses Palmer's Penguins
        full_df = sns.load_dataset("penguins").drop(columns = ["island", "sex"]).dropna()
        y = pd.factorize(full_df["species"])[0]
        labels = full_df["species"]
        target_names = labels.unique()
    else: # If user chooses their own dataset
        labels = None

    # PCA Scatterplot
    st.subheader("📊 PCA Scatterplot")
    plt.figure(figsize = (8, 6))

    if labels is not None:
        colors = ['navy', 'darkorange', 'green']
        for i, target_name in enumerate(target_names):
            plt.scatter(
                components[y == i, 0], components[y == i, 1],
                alpha = 0.7, edgecolor = 'k', s = 50,
                label = target_name, color = colors[i % len(colors)]
            )
        plt.legend(loc = "best")
    else:
        plt.scatter(components[:, 0], components[:, 1], alpha = 0.7, edgecolor = 'k', s = 50)

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Scatterplot (2D Projection)")
    plt.grid(True)
    st.pyplot(plt)

    # PCA Biplot
    st.subheader("📌 PCA Biplot (Demonstrating Features' Influence)")
    plt.figure(figsize = (8, 6))
    plt.scatter(components[:, 0], components[:, 1], alpha = 0.2, edgecolor = 'gray', s = 50)
    for i, feature in enumerate(selected_features):
        plt.arrow(0, 0, pca.components_[0, i]*3, pca.components_[1, i]*3,
                  color = 'r', alpha = 0.7, head_width = 0.1)
        plt.text(pca.components_[0, i]*3.2, pca.components_[1, i]*3.2, feature, color = 'r')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Biplot")
    plt.grid(True)
    st.pyplot(plt)

    # Scree Plot: Cumulative Explained Variance
    st.subheader("📉 Scree Plot (Cumulative Explained Variance)")
    pca_full = PCA(n_components = min(15, X_scaled.shape[1])).fit(X_scaled)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    plt.figure(figsize = (8, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker = 'o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Variance Explained')
    plt.xticks(range(1, len(cumulative_variance) + 1))
    plt.grid(True)
    st.pyplot(plt)

    st.divider()

    # Variance Explained
    st.markdown("## 🧮 Variance Explained")
    st.write(f"**Principal Component 1 explains:** {explained_var[0]:.2%} of variance")
    st.write(f"**Principal Component 2 explains:** {explained_var[1]:.2%} of variance")
    st.write(f"**Cumulative:** {cumulative_var[1]:.2%} of variance")