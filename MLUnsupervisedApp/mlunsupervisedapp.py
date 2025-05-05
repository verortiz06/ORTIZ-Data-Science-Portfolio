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
# Unsupervised ML explanation
st.markdown("## 📋 About This App:")
st.markdown("**Unsupervised machine learning** is a type of machine learning where we look for patterns and structure "
"in data *without* being given specific labels or outcomes to predict. It's like grouping similar items " \
"together without being told what the groups should be!")
# App description and explanation
st.markdown("This interactive application allows you to upload your own dataset, learn about different " \
"methods of unsupervised machine learning, experiment with hyperparameters, and observe how you can affect " \
"the model's training and performance.")

st.info("Let's build a machine learning model!")


# -----------------------------------------------
# Step 1: Choose Dataset
# -----------------------------------------------
st.sidebar.header("Step 1: Upload or Select Dataset")
sample_datasets = {
    "Iris Dataset": sns.load_dataset("iris"),
    "Palmer's Penguins": sns.load_dataset("penguins").drop(columns = ["island", "sex"]).dropna() # Drop categorical cols and missing values
}

dataset_choice = st.sidebar.selectbox("✨Choose Dataset✨", ["Upload Your Own"] + list(sample_datasets))
if dataset_choice == "Upload Your Own":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type = ["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file.")
        st.stop()
else:
    df = sample_datasets[dataset_choice]


# -----------------------------------------------
# Dataset Information in Sidebar
# -----------------------------------------------
st.sidebar.markdown("---") # Add divider
st.sidebar.subheader("Instant Dataset Info")
st.sidebar.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
st.sidebar.markdown("---")

st.divider()


# -----------------------------------------------
# Dataset Preview
# -----------------------------------------------
st.write("### 🔍 Dataset Preview")
st.dataframe(df.head())
st.markdown("""
Here you can see the first five rows of any dataset you've selected or uploaded!
* **Rows:** each row represents a single observation or data point
* **Columns:** each column represents a specific feature or characteristic of that data point
""")

st.divider()


# -----------------------------------------------
# Step 2: Feature Selection + Scaling
# -----------------------------------------------
st.sidebar.header("Step 2: Choose Features")
st.sidebar.markdown("""
In this step, you can choose which characteristics (or columns) from the dataset you'd want your 
model to include in its search for a pattern!
""")
numeric_cols = df.select_dtypes(include = [np.number]).columns.tolist()
selected_features = st.sidebar.multiselect("✨Select Features:✨", numeric_cols, default = numeric_cols)
st.sidebar.markdown("""
After you've selected your features, we'll automatically **scale** the data using standard deviations.

This is necessary because models such as K-Means and Hierarchical clustering calculate the distances between
data points, which means that a uniform scale between the features is important.
""")
st.sidebar.markdown("---")

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
model_choice = st.sidebar.selectbox("✨Choose a Model Type:✨", ["K-Means Clustering", "Hierarchical Clustering", "PCA"])
if model_choice == "K-Means Clustering":
    st.sidebar.markdown("""
    **K-Means Clustering** groups data into *k* clusters based on feature similarity. 
                        
    It iteratively assigns a point to the nearest cluster "centroid", which is initially randomly placed, 
    and then updates the positions of the centroids based on the mean of the clusters created.
    """)
    st.sidebar.markdown("Learn more about [K-Means Clustering](https://www.geeksforgeeks.org/k-means-clustering-introduction/)!")
if model_choice == "Hierarchical Clustering":
    st.sidebar.markdown("""
    **Hierarchical Clustering** creates a dendogram, which is a sort of hierarchical tree
    that demonstrates how the data are related a different levels.
                        
    We will be using the *ward* linkage method, which means that clusters will be merged in a way that 
    results in the *smallest* increase of within-cluster variance.
                        """)
    st.sidebar.markdown("Learn more about [Hierarchical Clustering](https://www.geeksforgeeks.org/hierarchical-clustering/)!")
if model_choice == "PCA":
    st.sidebar.markdown("""
    **PCA (or Principal Component Analysis)** is a method in which you can reduce dimensionality to 2 dimensions. 
                        
    These linear combinations are axes, or *principal components*. We reduce down to 2 components for visualization 
    purposes, which can illustrate for us the influence of the original features.
                        """)
    st.sidebar.markdown("Learn more about [Principal Component Analysis](https://www.geeksforgeeks.org/principal-component-analysis-pca/)")
st.sidebar.markdown("---")


# -----------------------------------------------
# Hyperparameters, Model Training, and Visualizations
# -----------------------------------------------
if model_choice == "K-Means Clustering":
    
    # Step 4: Choosing Hyperparameters
    st.sidebar.header("Step 4: Choose Hyperparameters")
    st.sidebar.markdown("**Hyperparameters** are a type of setting that you can prescribe to the model before it starts learning from the data.")
    k = st.sidebar.slider("✨Number of Clusters (k)✨", 2, 10, 3)
    st.sidebar.markdown("For K-Means, the main hyperparameter is **k**, which represents the number of clusters." \
    "Essentially, you are telling the model how many groups you are wanting it to find in the data. Experiment with this to see how it" \
    "affects the clustering!")
    
    # Training Model
    model = KMeans(n_clusters = k, random_state = 20)
    labels = model.fit_predict(X_scaled)
    
    # Cluster scatterplot
    st.markdown("## Model Visualizations:")
    st.subheader("📊 Cluster Scatterplot")
    st.markdown("Since your original data might have many features (or dimensions), we use **Principal Component Analysis (PCA)** here for " \
    "visualization purposes to reduce the data down to 2 main components (PC1 and PC2) that capture the most important patterns.")
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
    st.markdown("""
    * Each point is one data point from your dataset
    * The **color** of each point indicates the cluster that the K-Means model assigned it to""")

    # True labels comparison
    if "species" in df.columns:
        st.markdown("---")
        st.subheader("🎯 Comparing Clusters with True Labels")
        st.markdown("If your dataset happens to have known categories (like the 'species' in the Iris or Penguins datasets), we can compare the " \
        "clusters found by the algorithm to the actual known categories.")
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
        st.markdown("""
        * This plot shows the same 2D PCA projection as above, but this time the points are colored by their *true*, known category.
        * Compare this plot to the cluster scatterplot above. How well do the algorithm's clusters match the true categories?""")

    #st.markdown("## Model Performance Metrics:")


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
    st.markdown("Choosing the best number of clusters (*k*) can be tricky. Here are two common methods that can help" \
    " you decide visually:")

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
    st.markdown("""
    The **Elbow Method** uses the **WCSS (Within-Cluster Sum of Squares)** to help find a good value for *k*.
    * The plot above shows the WCSS calculated for different numbers of clusters (*k*). As *k* increases, WCSS will naturally decrease (because 
    you're splitting the data into more, smaller groups, so points will be closer to their assigned center).
    * The 'elbow' is the point on the graph where the rate of decrease in WCSS sharply changes, looking like an elbow joint. This point is often 
    considered a good candidate for the optimal *k* because adding more clusters beyond this point doesn't give you a significant reduction in WCSS.
    """)

    # Silhouette Plot
    plt.figure()
    plt.plot(list(k_range), sil_scores, marker = 'o', color = 'orange')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score for Optimal k")
    st.pyplot(plt)

    # Silhouette Score
    silhouette = silhouette_score(X_scaled, labels)
    st.metric("Silhouette Score", f"{silhouette:.3f}")

    st.markdown("""
    The **Silhouette Score** is a metric used to evaluate the quality of clusters. It measures how similar a data point is to its own cluster compared to other clusters.

    * The score ranges from -1 to +1.
    * A score close to **+1** means the data point is well within its own cluster and far from other clusters (good clustering).
    * A score close to **0** means the data point is near the boundary between two clusters.
    * A score close to **-1** means the data point might have been assigned to the wrong cluster.

    The value shown below is the average Silhouette Score across all data points. A higher average score generally indicates better clustering.
    """)

    # Optimal k
    st.info(f"Best number of clusters by silhouette score: **{best_k}** (score = {best_score:.3f})")

elif model_choice == "Hierarchical Clustering":
    
    # Step 4: Choose Hyperparameters
    st.sidebar.header("Step 4: Choose Hyperparameters:")
    st.sidebar.markdown("**Hyperparameters** are a type of setting that you can prescribe to the model before it starts learning from the data.")
    n_clusters = st.sidebar.slider("✨Number of Clusters (k)✨", 2, 10, 3)
    st.sidebar.markdown("""For Hierarchical Clustering, the main hyperparameter is **k**, which represents the number of clusters you want to extract 
    from the dendogram. Choosing the k value means that you are deciding how many final groups you want. 
    Conceptually, you are cutting down the tree by doing this.""")

    # Training Model
    method = "ward"
    model = AgglomerativeClustering(n_clusters = n_clusters, linkage = "ward")
    labels = model.fit_predict(X_scaled)

    # Cluster scatterplot
    st.markdown("## Model Visualizations:")
    st.subheader("📊 Cluster Scatterplot (via PCA)")
    st.markdown("Similar to the K-Means visualization, this plot shows your data points in a 2-dimensional space using **Principal Component Analysis (PCA)** for visualization.")
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
    st.markdown("""
    * Each dot is a data point
    * The **color** indicates the cluster assigned by the Hierarchical Clustering model based on the number of clusters (*k*) you selected""")

    # True labels comparison
    if "species" in df.columns:
        st.markdown("---")
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
        st.markdown("If your dataset happens to have known categories (like the 'species' in the Iris or Penguins datasets), we can compare the " \
        "clusters found by the algorithm to the actual known categories.")

    # Dendrogram
    st.markdown("---")
    st.subheader("🌳 Dendrogram")
    st.markdown("The **Dendrogram** is the main output of Hierarchical Clustering! It's a tree diagram that illustrates the sequence of merges or splits of clusters.")
    Z = linkage(X_scaled, method = method)
    plt.figure(figsize = (10, 5))
    dendrogram(Z)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample Index") # and how on earth do i make this look better
    plt.ylabel("Distance") 
    st.pyplot(plt)
    st.markdown("""
    * The **leaves** at the bottom represent individual data points.
    * The **branches** show how data points are grouped together into clusters.
    * The **distance** of the merge points on the vertical axis represents the dissimilarity between the clusters being merged.
    """)
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

    # Silhouette Score Plot
    plt.figure(figsize = (7, 4))
    plt.plot(list(k_range), sil_scores, marker = "o", color = 'teal')
    plt.xticks(list(k_range))
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Average Silhouette Score")
    plt.title("Silhouette Analysis for Agglomerative (Ward) Clustering")
    plt.grid(True, alpha = 0.3)
    st.pyplot(plt)


    # Silhouette Score
    silhouette = silhouette_score(X_scaled, labels)
    st.metric("Silhouette Score", f"{silhouette:.3f}")
    st.markdown("""
    We can use the **Silhouette Score** to help evaluate the quality of clustering for different numbers of clusters (*k*) when using Hierarchical Clustering.

    This plot shows the average Silhouette Score for different values of *k* (from 2 to 10). Look for the peak in the graph - the value of *k* with the highest score is suggested as a good number of clusters for this data and method.
    """)

    # Optimal k
    st.info(f"Best number of clusters by silhouette score: **{best_k}** (score = {max(sil_scores):.3f})")

elif model_choice == "PCA":
    st.markdown("## Model Visualizations:")
    st.markdown("**Principal Component Analysis (PCA)** helps us understand and visualize data with many features by reducing the number of dimensions, yet still keeping the most important information.")

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
    st.markdown("This plot shows your data points as the first two **Principal Components** (PC1 and PC2). These components are new 'features' that are actually created by the PCA model itself " \
    "that capture the most variance from the original data.")

    # PCA Biplot
    st.markdown("---")
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
    st.markdown("""
    A **Biplot** combines the PCA scatterplot with arrows showing the direction and strength of the original features in the new 2D space.
    * The **red arrows** represent the original features you selected in the sidebar
    * The **direction** of an arrow shows how that feature contributes to the principal components. Features pointing in similar directions are likely correlated
    * The **length** of an vectors that indicate the strength of a feature's influence on the principal components shown (PC1 and PC2)
                """)

    # Scree Plot: Cumulative Explained Variance
    st.markdown("---")
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
    st.markdown("""
    The **Scree Plot** helps you understand how much of the total information (variance) in your original data is captured by each principal component.
    * The plot shows the **cumulative explained variance** as you add more principal components
    * Ideally, you want to use a number of components that capture a high percentage of the total variance without being too many. The plot often shows an 
    'elbow' where adding more components fails to give you much more variance    
               """)

    st.divider()

    # Variance Explained
    st.markdown("## 🧮 Variance Explained by Model:")
    st.markdown("Here's how much of the total information (variance) from your original selected features is captured by the first two principal components:")
    st.markdown(f"**Principal Component 1 explains:** {explained_var[0]:.2%} of variance")
    st.markdown(f"**Principal Component 2 explains:** {explained_var[1]:.2%} of variance")
    st.markdown(f"**Cumulative:** {cumulative_var[1]:.2%} of variance")

# -----------------------------------------------
# More Resources for Learning
# -----------------------------------------------
st.sidebar.divider()
st.sidebar.markdown("""
**Interested in learning EVEN more?** 
                    
Here are some extra resources to fuel your curiousity!
                    
* [Machine Learning 101](https://www.geeksforgeeks.org/machine-learning/)
* [Unsupervised Learning Overview](https://www.geeksforgeeks.org/unsupervised-learning/)
* [Data Scaling](https://www.geeksforgeeks.org/ml-feature-scaling-part-2/)
""")