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
    "Iris Dataset": sns.load_dataset("iris").dropna(), # first datset option: Iris
    "Palmer's Penguins": sns.load_dataset("penguins").dropna() # second datset option: Penguins; Drop categorical cols and missing values
}

dataset_choice = st.sidebar.selectbox("✨Choose Dataset✨", ["Upload Your Own"] + list(sample_datasets)) # user can choose their own dataset 
if dataset_choice == "Upload Your Own":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type = ["csv"]) # must be in a csv format
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        # Preprocessing uploaded data:
        # Drop rows with any missing values
        initial_rows = df.shape[0]
        df = df.dropna()
        rows_after_dropna = df.shape[0]
        if initial_rows > rows_after_dropna:
            st.warning(f"Dropped {initial_rows - rows_after_dropna} row(s) containing missing values.")
        # Check if DataFrame is empty after dropping the missing values
        if df.empty:
            st.error("The dataset is empty after removing rows with missing values. Please upload a dataset with valid data.")
            st.stop() # stops the app since dataset has been emptied
        # Convert categorical variables to dummy variables
        df_processed = pd.get_dummies(df)
        df = df_processed # updates the DataFrame to the processed one
    else:
        st.warning("Please upload a CSV file.") # if they do not upload a dataset, the app tells them to or else nothing will happen
        st.stop()
else:
    df = sample_datasets[dataset_choice]

# -----------------------------------------------
# Prepare True Labels for Comparison (if 'species' exists)
# -----------------------------------------------
# Initialize variables to store true labels for comparison plots
y_true_comparison = None
target_names_comparison = None
has_true_labels = False

# Check if the 'species' column exists in the dataset AFTER initial loading/cleaning
if "species" in df.columns:
    has_true_labels = True
    true_labels = df["species"] # Get the 'species' column
    target_names_comparison = true_labels.unique() # Get the unique names
    # Create a numerical mapping for plotting colors
    label_map = {name: idx for idx, name in enumerate(target_names_comparison)}
    y_true_comparison = true_labels.map(label_map) # Convert labels to numerical format


# -----------------------------------------------
# Dataset Information in Sidebar
# -----------------------------------------------
st.sidebar.markdown("---") # Add divider
st.sidebar.subheader("Instant Dataset Info") 
st.sidebar.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns") # displays the number of rows and columns of the dataset they choose
st.sidebar.markdown("---")

st.divider()


# -----------------------------------------------
# Dataset Preview
# -----------------------------------------------
st.write("### 🔍 Dataset Preview")
st.dataframe(df.head()) # shows the first five rows of the dataset
st.markdown(""" 
Here you can see the first five rows of any dataset you've selected or uploaded!
* **Rows:** each row represents a single observation or data point
* **Columns:** each column represents a specific feature or characteristic of that data point
""") # explanation for the preview

st.divider()


# -----------------------------------------------
# Step 2: Feature Selection + Scaling
# -----------------------------------------------
st.sidebar.header("Step 2: Choose Features")
st.sidebar.markdown("""
In this step, you can choose which characteristics (or columns) from the dataset you'd want your 
model to include in its search for a pattern!
""")

all_cols = df.columns.tolist() # gets list of all columns available for selection
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
selected_features = st.sidebar.multiselect(
    "✨Select Features:✨",
    all_cols, # Provide all columns as options
    default = all_cols # set numeric columns as the default selection
)
st.sidebar.markdown("""
After you've selected your features, we will automatically:
1. Convert selected categorical features into numerical **dummy variables**
2. **Scale** all the selected (and converted) data using standard deviations

Scaling is necessary because models such as K-Means and Hierarchical clustering calculate the distances between
data points, which means that a uniform scale between the features is important.
""") # explanation of the step for the user
st.sidebar.markdown("---")

if len(selected_features) < 2: # checks if at least two original features were selected
    st.warning("Please select at least two features to proceed.")
    st.stop()

df_selected = df[selected_features] # creating dataframe with only the selected features
X = pd.get_dummies(df_selected) # converting selected categorical features to dummy variables
scaler = StandardScaler() # scales the features using standard deviations so that the distance being calculated between them is uniform
X_scaled = scaler.fit_transform(X)


# -----------------------------------------------
# Step 3: Choose Model
# -----------------------------------------------
st.sidebar.header("Step 3: Choose a Model")
model_choice = st.sidebar.selectbox("✨Choose a Model Type:✨", ["K-Means Clustering", "Hierarchical Clustering", "PCA"]) # allows the user to choose from three different models for the machine learning
if model_choice == "K-Means Clustering": # the information that will pop-up if the user chooses a K-Means clustering model
    st.sidebar.markdown("""
    **K-Means Clustering** groups data into *k* clusters based on feature similarity. 
                        
    It iteratively assigns a point to the nearest cluster "centroid", which is initially randomly placed, 
    and then updates the positions of the centroids based on the mean of the clusters created.
    """)
    st.sidebar.markdown("Learn more about [K-Means Clustering](https://www.geeksforgeeks.org/k-means-clustering-introduction/)!") # external link where the user can learn more about K-Means
if model_choice == "Hierarchical Clustering": # the information that will pop-up if the user chooses a Hierarchical clustering model
    st.sidebar.markdown("""
    **Hierarchical Clustering** creates a dendogram, which is a sort of hierarchical tree
    that demonstrates how the data are related a different levels.
                        
    We will be using the *ward* linkage method, which means that clusters will be merged in a way that 
    results in the *smallest* increase of within-cluster variance.
                        """)
    st.sidebar.markdown("Learn more about [Hierarchical Clustering](https://www.geeksforgeeks.org/hierarchical-clustering/)!") # external link where the user can learn more about Hierarchical clustering
if model_choice == "PCA": # the information that will pop-up if the user chooses PCA model
    st.sidebar.markdown("""
    **PCA (or Principal Component Analysis)** is a method in which you can reduce dimensionality to 2 dimensions. 
                        
    These linear combinations are axes, or *principal components*. We reduce down to 2 components for visualization 
    purposes, which can illustrate for us the influence of the original features.
                        """)
    st.sidebar.markdown("Learn more about [Principal Component Analysis](https://www.geeksforgeeks.org/principal-component-analysis-pca/)") # external link where the user can learn more about PCA


# -----------------------------------------------
# Step 4: Hyperparameters, plus Model Training, and Visualizations
# -----------------------------------------------

# -----------------------------------------------
# K-Means
# -----------------------------------------------
if model_choice == "K-Means Clustering": # If the user chooses a K-Means machine learning model
    
    # Step 4: Choosing Hyperparameters
    st.sidebar.header("Step 4: Choose Hyperparameters")
    st.sidebar.markdown("**Hyperparameters** are a type of setting that you can prescribe to the model before it starts learning from the data.") # explanation
    k = st.sidebar.slider("✨Number of Clusters (k)✨", 2, 10) # allows the user to choose a number of clusters between 2 and 10
    st.sidebar.markdown("For K-Means, the main hyperparameter is **k**, which represents the number of clusters." \
    "Essentially, you are telling the model how many groups you are wanting it to find in the data. Experiment with this to see how it" \
    "affects the clustering!") # sidebar explanation for what clusters (k) are
    
    # Training Model
    model = KMeans(n_clusters = k, random_state = 20) # initializing the k-means model training
    labels = model.fit_predict(X_scaled) # this gets the cluster label for each data point
    
    # Cluster scatterplot
    st.markdown("## Model Visualizations:") 
    st.subheader("📊 Cluster Scatterplot")
    st.markdown("Since your original data might have many features (or dimensions), we use **Principal Component Analysis (PCA)** here for " \
    "visualization purposes to reduce the data down to 2 main components (PC1 and PC2) that capture the most important patterns.") # cluster scatterplot explanation
    pca = PCA(n_components = 2) # use PCA to reduce the scaled data into 2 dimensions
    pca_components = pca.fit_transform(X_scaled) # applies PCA to the scaled data
    plt.figure()
    scatter = plt.scatter(
        pca_components[:, 0], # x-axis; first principal component
        pca_components[:, 1], # y-axis; second principal component
        c = labels, # colors the data points based on the cluster
        cmap = "Accent", 
        s = 50)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("K-Means Cluster Visualization")
    plt.grid(True, alpha = 0.3) # adds a grid to the plot and some transparency
    plt.legend(*scatter.legend_elements(), title = "Clusters")  # Legend for the clusters
    st.pyplot(plt)
    st.markdown("""
    * Each point is one data point from your dataset
    * The **color** of each point indicates the cluster that the K-Means model assigned it to""") # scatterplot explanation

    # True labels comparison
    if has_true_labels: # true label comparison only shows if there is a "species" feature in the dataset
        st.markdown("---")
        st.subheader("🎯 Comparing Clusters with True Labels")
        st.markdown("If your dataset happens to have known categories (like the 'species' in the Iris or Penguins datasets), we can compare the " \
        "clusters found by the algorithm to the actual known categories.")
        true_labels = df["species"] # getting the true labels from the original dataset
        target_names = true_labels.unique()
        label_map = {name: idx for idx, name in enumerate(target_names)} # creating a map from category names to numerical indices
        y_true = true_labels.map(label_map) # applying that map to convert true labels to numerical indices

        plt.figure(figsize = (8, 6))
        for i, target_name in enumerate(target_names_comparison): # loop through each unique true label
            plt.scatter(
                pca_components[y_true_comparison == i, 0], # x-axis; PC1 values for the true labels
                pca_components[y_true_comparison == i, 1], # y-axis; PC2 values for the true labels
                alpha = 0.7,
                edgecolor = 'k',
                label = target_name,
                s = 50
            )
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("True Labels: 2D PCA Projection")
        plt.legend(loc = "best") # adds a legend for the true labels
        plt.grid(True, alpha = 0.3) # adds a grid to the plot and some transparency
        st.pyplot(plt)
        st.markdown("""
        * This plot shows the same 2D PCA projection as above, but this time the points are colored by their *true*, known category
        * Compare this plot to the cluster scatterplot above. How well do the algorithm's clusters match the true categories?""") # true clusters explanation
    
    st.divider()

    # -----------------------------------------------
    # Evaluating Optimal Number of Clusters
    # -----------------------------------------------
    # Elbow Method (WCSS) and Silhouette Score Plots
    st.markdown("## Evaluating Optimal Number of Clusters:")
    st.markdown("Choosing the best number of clusters (*k*) can be tricky. Here are two common methods that can help" \
    " you decide visually:") # intro explanation

    st.subheader("📉 Elbow Method and Silhouette Score")
    sse = [] # list to store WCSS
    sil_scores = [] # list to store Silhouette Scores
    k_range = range(2, 11)  # silhouette scores are only valid for k >= 2

    for k_val in k_range:
        kmeans = KMeans(n_clusters = k_val, random_state = 20) # initialize k-means and fit for each k value
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)
        score = silhouette_score(X_scaled, kmeans.labels_) # calculate the silhouette score for the chosen clustering and append
        sil_scores.append(score)
    
    # Find the k value that resulted in the maximum silhouette score
    best_k = k_range[np.argmax(sil_scores)]
    best_score = max(sil_scores)

    # Elbow Plot
    plt.figure()
    plt.plot(list(k_range), sse, marker = 'o') # plot WCSS against the number of clusters
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
    plt.title("Elbow Plot (WCSS) for Optimal k")
    st.pyplot(plt)
    st.markdown("""
    The **Elbow Method** uses the **WCSS (Within-Cluster Sum of Squares)** to help find a good value for *k*
    * The plot above shows the WCSS calculated for different numbers of clusters (*k*). As *k* increases, WCSS will naturally decrease (because 
    you're splitting the data into more, smaller groups, so points will be closer to their assigned center)
    * The 'elbow' is the point on the graph where the rate of decrease in WCSS sharply changes, looking like an elbow joint. This point is often 
    considered a good candidate for the optimal *k* because adding more clusters beyond this point doesn't give you a significant reduction in WCSS
    """) # elbow plot explanation

    # Silhouette Plot
    plt.figure()
    plt.plot(list(k_range), sil_scores, marker = 'o', color = 'orange') # plot Silhouette Score against the number of clusters
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score for Optimal k")
    st.pyplot(plt)

    # Silhouette Score
    silhouette = silhouette_score(X_scaled, labels) # calculate the exact Silhouette Score for the specific situation the user has chosen
    st.metric("Silhouette Score", f"{silhouette:.3f}")

    st.markdown("""
    The **Silhouette Score** is a metric used to evaluate the quality of clusters. It measures how similar a data point is to its own cluster compared to other clusters.

    * The score ranges from -1 to +1
    * A score close to **+1** means the data point is well within its own cluster and far from other clusters (good clustering)
    * A score close to **0** means the data point is near the boundary between two clusters
    * A score close to **-1** means the data point might have been assigned to the wrong cluster

    The value shown below is the average Silhouette Score across all data points. A higher average score generally indicates better clustering.
    """) # Silhouette Score explanation

    # Optimal k based on user's selections
    st.info(f"Best number of clusters by silhouette score: **{best_k}** (score = {best_score:.3f})")


# -----------------------------------------------
# Hierarchical Clustering
# -----------------------------------------------
elif model_choice == "Hierarchical Clustering": # If the user chooses a Hierarchical Clustering machine learning model
    
    # Step 4: Choose Hyperparameters
    st.sidebar.header("Step 4: Choose Hyperparameters:") 
    st.sidebar.markdown("**Hyperparameters** are a type of setting that you can prescribe to the model before it starts learning from the data.") # hierarchical clustering explanation
    n_clusters = st.sidebar.slider("✨Number of Clusters (k)✨", 2, 10) # allows the user to choose a number of clusters between 2 and 10
    st.sidebar.markdown("""For Hierarchical Clustering, the main hyperparameter is **k**, which represents the number of clusters you want to extract 
    from the dendogram. Choosing the k value means that you are deciding how many final groups you want. 
    Conceptually, you are cutting down the tree by doing this.""") # hierarchical clustering explanation

    # Training Model
    method = "ward" # using this specific linkage method
    model = AgglomerativeClustering(n_clusters = n_clusters, linkage = "ward") # initializing the model
    labels = model.fit_predict(X_scaled) # fit the model to the scaled data and predict the cluster label for each data point

    # Cluster scatterplot
    st.markdown("## Model Visualizations:")
    st.subheader("📊 Cluster Scatterplot (via PCA)")
    st.markdown("Similar to the K-Means visualization, this plot shows your data points in a 2-dimensional space using **Principal Component Analysis (PCA)** for visualization.") # cluster scatterplot explanation
    pca = PCA(n_components = 2) # use PCA to reduce the scaled data into 2 dimensions
    reduced = pca.fit_transform(X_scaled) # applies PCA to the scaled data
    plt.figure()
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c = labels, cmap = 'plasma') # plots the data points as PC1 and PC2 in a 2-D graph
    plt.legend(*scatter.legend_elements(), title = "Clusters") # adds a legend for the clusters
    plt.title("Hierarchical Clustering (Method: ward)")
    plt.grid(True, alpha = 0.3) # adds grid lines to plot and some transparency
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    st.pyplot(plt)
    st.markdown("""
    * Each dot is a data point
    * The **color** indicates the cluster assigned by the Hierarchical Clustering model based on the number of clusters (*k*) you selected""") # cluster scatterplot explanation

    # True labels comparison
    if has_true_labels: # true label comparison only shows if there is a "species" feature in the dataset
        st.markdown("---")
        st.subheader("🎯 Comparing Clusters with True Labels")
        true_labels = df["species"] # getting the true labels from the original dataset
        target_names = true_labels.unique()
        label_map = {name: idx for idx, name in enumerate(target_names)} # creating a map from category names to numerical indices
        y_true = true_labels.map(label_map) # applying that map to convert true labels to numerical indices

        plt.figure(figsize = (8, 6))
        for i, target_name in enumerate(target_names_comparison): # loop through each unique true label
            plt.scatter(
                reduced[y_true_comparison == i, 0], # x-axis; PC1 values for the true labels
                reduced[y_true_comparison == i, 1], # y-axis; PC2 values for the true labels
                alpha = 0.7,
                edgecolor = 'k',
                label = target_name,
                s = 50
            )
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("True Labels: 2D PCA Projection")
        plt.legend(loc = "best") # adds a legend for the true labels
        plt.grid(True, alpha = 0.3) # adds a grid to the plot and some transparency
        st.pyplot(plt)
        st.markdown("If your dataset happens to have known categories (like the 'species' in the Iris or Penguins datasets), we can compare the " \
        "clusters found by the algorithm to the actual known categories.") # plot explanation

    # Dendrogram
    st.markdown("---")
    st.subheader("🌳 Dendrogram")
    st.markdown("The **Dendrogram** is the main output of Hierarchical Clustering! It's a tree diagram that illustrates the sequence of merges or splits of clusters.") # dendrogram explanation
    Z = linkage(X_scaled, method = method) # do the hierarchical clustering linkage calculation using 'ward' method on the scaled data
    plt.figure(figsize = (10, 5))
    dendrogram(Z,
               truncate_mode = "lastp") # truncates the number of examples shown
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample Index") # and how on earth do i make this look better
    plt.ylabel("Distance") 
    st.pyplot(plt)
    st.markdown("""
    * The **leaves** at the bottom represent individual data points
    * The **branches** show how data points are grouped together into clusters
    * The **distance** of the merge points on the vertical axis represents the dissimilarity between the clusters being merged
    * Since showing *every* single data point in the dendrogram can look overwheling and unflattering, we **truncate** it so that the number of data points belong to the unique branches
    are displayed in parantheses to represent a merged cluster
    """) # dendrogram plot explanation

    st.divider()

    # -----------------------------------------------
    # Evaluating Optimal Number of Clusters
    # -----------------------------------------------
    # Silhouette Elbow Plot
    st.markdown("## Evaluating Optimal Number of Clusters:")
    k_range = range(2, 11) # range of k values to be tested
    sil_scores = [] # list to store the Silhouette Scores

    for k in k_range:
        hc = AgglomerativeClustering(n_clusters = k, linkage = 'ward') # initialize clustering for each k value, re-fitting each time
        labels_k = hc.fit_predict(X_scaled) # getting the cluster labels
        score = silhouette_score(X_scaled, labels_k) # calculate the Silhouette Score using the scaled data and labels
        sil_scores.append(score)

    # Finding the optimal k value based on maximum Silhouette Score 
    best_k = k_range[np.argmax(sil_scores)]

    # Silhouette Score Plot
    plt.figure(figsize = (7, 4))
    plt.plot(list(k_range), sil_scores, marker = "o", color = 'teal') #
    plt.xticks(list(k_range)) # adds integer ticks on the sides of the plot
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Average Silhouette Score")
    plt.title("Silhouette Analysis for Agglomerative (Ward) Clustering")
    plt.grid(True, alpha = 0.3) # adds grid lines to the plot and some transparency
    st.pyplot(plt)


    # Silhouette Score
    silhouette = silhouette_score(X_scaled, labels) # calculate the exact Silhouette Score for the specific situation the user has chosen
    st.metric("Silhouette Score", f"{silhouette:.3f}")
    st.markdown("""
    We can use the **Silhouette Score** to help evaluate the quality of clustering for different numbers of clusters (*k*) when using Hierarchical Clustering.

    This plot shows the average Silhouette Score for different values of *k* (from 2 to 10). Look for the peak in the graph - the value of *k* with the highest score is suggested as a good number of clusters for this data and method.
    """) # Silhouette Score explanation

    # Optimal k based on user's selections
    st.info(f"Best number of clusters by silhouette score: **{best_k}** (score = {max(sil_scores):.3f})")


# -----------------------------------------------
# PCA
# -----------------------------------------------
elif model_choice == "PCA": # If the user chooses a PCA machine learning model
    st.markdown("## Model Visualizations:")
    st.markdown("**Principal Component Analysis (PCA)** helps us understand and visualize data with many features by reducing the number of dimensions, yet still keeping the most important information.") # PCA explanation

    # Initalize PCA
    pca = PCA(n_components = 2) # initialize PCA and reduce data to 2 components
    components = pca.fit_transform(X_scaled) # applies PCA to the scaled data
    explained_var = pca.explained_variance_ratio_ # get the variance explained by each of the first two components
    cumulative_var = np.cumsum(explained_var) # calculate the total variance explained by both components

    # PCA Scatterplot
    st.subheader("📊 PCA Scatterplot")
    plt.figure(figsize = (8, 6))

    if has_true_labels: # plotting true labels by color if available
        colors = ['navy', 'darkorange', 'green']
        for i, target_name in enumerate(target_names_comparison): # loop through each true label (species)
            plt.scatter(
                components[y_true_comparison == i, 0], # x-axis; PC1
                components[y_true_comparison == i, 1], #y-axis; PC2
                alpha = 0.7, edgecolor = 'k', s = 50,
                label = target_name, color = colors[i % len(colors)] # labels for legend and cycle through colors
            )
        plt.legend(loc = "best")
    else:
        plt.scatter(components[:, 0], components[:, 1], alpha = 0.7, edgecolor = 'k', s = 50)

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Scatterplot (2D Projection)")
    plt.grid(True, alpha = 0.3) # adds grid lines to the plot and some transparency
    st.pyplot(plt)
    st.markdown("This plot shows your data points as the first two **Principal Components** (PC1 and PC2). These components are new 'features' that are actually created by the PCA model itself " \
    "that capture the most variance from the original data.") # scatterplot explanation

    # PCA Biplot
    st.markdown("---")
    st.subheader("📌 PCA Biplot (Demonstrating Features' Influence)")
    plt.figure(figsize = (8, 6))
    plt.scatter(components[:, 0], # PC1
                components[:, 1], # PC2
                alpha = 0.2, edgecolor = 'gray', s = 50) # leave dots on graph gray to emphazie the arrows
    for i, feature in enumerate(selected_features): # adding arrows representing the original features
        plt.arrow(0, 0, 
                  pca.components_[0, i]*3, 
                  pca.components_[1, i]*3,
                  color = 'r', alpha = 0.7, head_width = 0.1) # arrow properties
        plt.text(pca.components_[0, i]*3.2, pca.components_[1, i]*3.2, feature, color = 'r') # adjusting the text
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Biplot")
    plt.grid(True, alpha = 0.3) # adds grid lines to the plot and some transparency
    st.pyplot(plt)
    st.markdown("""
    A **Biplot** combines the PCA scatterplot with arrows showing the direction and strength of the original features in the new 2D space.
    * The **red arrows** represent the original features you selected in the sidebar
    * The **direction** of an arrow shows how that feature contributes to the principal components. Features pointing in similar directions are likely correlated
    * The **length** of an vectors that indicate the strength of a feature's influence on the principal components shown (PC1 and PC2)
                """) # biplot explanation

    # Combined Variance Plot
    st.markdown("---")
    st.subheader("📉 Combined Variance Plot")
    n_components_to_plot = min(X_scaled.shape[1], 15)
    pca_full = PCA(n_components = n_components_to_plot).fit(X_scaled)
    # Get explained variance ratios and converting them to percentages for the plot
    individual_variance_ratio = pca_full.explained_variance_ratio_
    individual_variance_percent = individual_variance_ratio * 100 
    cumulative_variance_ratio = np.cumsum(individual_variance_ratio)
    cumulative_variance_percent = cumulative_variance_ratio * 100 
    components = range(1, len(individual_variance_percent) + 1) # component numbers for x-axis
    fig, ax1 = plt.subplots(figsize=(10, 6)) # create the figure
    # Making the bar plot for INDIVIDUAL variance explained
    bar_color = 'steelblue'
    ax1.bar(components, individual_variance_percent, color = bar_color, alpha = 0.8, label = 'Individual Variance')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Individual Variance Explained (%)', color = bar_color)
    ax1.tick_params(axis = 'y', labelcolor = bar_color)
    ax1.set_xticks(components)
    ax1.set_xticklabels([f"PC{i}" for i in components]) # formatting x-axis labels as PC1, PC2, etc
    for i, v in enumerate(individual_variance_percent): # adding percentage labels on top of each bar
        ax1.text(components[i], v + 1, f"{v:.1f}%", ha = 'center', va = 'bottom', fontsize = 9, color = 'black')
    # Creating the second y-axis for CUMULATIVE variance explained
    ax2 = ax1.twinx()
    line_color = 'crimson' # making this other part of the graph look distinct from bar chart
    ax2.plot(components, cumulative_variance_percent, color = line_color, marker = 'o', label = 'Cumulative Variance')
    ax2.set_ylabel('Cumulative Variance Explained (%)', color = line_color)
    ax2.tick_params(axis = 'y', labelcolor = line_color)
    ax2.set_ylim(0, 100) # this sets the y-axis limit to 100%
    ax1.grid(False) # removing the grid lines to help readability
    ax2.grid(False)
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc = 'center right', bbox_to_anchor = (0.9, 0.5))
    plt.title('PCA: Variance Explained')
    st.pyplot(fig)
    st.markdown("""
    * The **blue bars** show the proportion of the total variance explained by each **individual** principal component (PC1, PC2, etc.). PC1 typically explains the most variance, and you can see that with additional
    components, the contribution to the explained variance begins decreasing dramatically
    * The **red line** shows the **cumulative** variance explained as you add more principal components. This line increases as you include more components, reaching 100% when all components are included
    * This plot is helpfup for deciding how many principal components to choose. A common strategy is to choose the number of components that explains a high percentage of the total variance while also considering the 
    "elbow" point where individual variance gain significantly decreases
    """) # combined variance plot explanation

    st.divider()

    # Variance Explained
    st.markdown("## 🧮 Variance Explained by Model:")
    st.markdown("Here's how much of the total information (variance) from your original selected features is captured by the first two principal components:") # explanation for what variance is
    # Displaying the component and cumulative variances:
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