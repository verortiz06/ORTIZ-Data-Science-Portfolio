# Streamlit App: Unsupervised Machine Learning
## 📌 Project Overview 
This project is centered around creating an interactive Streamlit app that allows users to explore core concepts of **Unsupervised Machine Learning** and pattern discovery in data! In this streamlit app, users can:
- Select from preloaded datasets (Iris or Palmer's Penguins) or upload their own `.csv` file
- Choose and play with different combinations of features from the dataset to be used by the machine learning model of their choice
- Choose from unsupervised machine learning models such as **K-Means Clustering, Hierarchical Clustering, and Principal Component Analysis (PCA)**
- Experiment with hyperparameters, such as the number of clusters (k), and see how the model's visualizatons, outputs, and overall performance changes
- Analyze the model's performance through model results and performance using various graphs, such as **cluster scatter plots, dendrograms, PCA biplots**, and **scree plots**
- Evaluate clustering quality using metrics like a model's **Silhouette Score** and the **Elbow Method (WCSS)**
- Compare clustering results to the sample datasets' true labels

## ⚙️ How to Run This Project
### 1️⃣ Ensure that you have imported Streamlit
### 2️⃣ Install Required Libraries and Packages
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the App
- Within your terminal, locate the correct location of the MLStreamlitApp file by using commands **"ls"** to explore your desktop and **"cd"** to "double-click" into your folders until you find *mlunsupervisedapp.py*
- Still within your terminal, type *"streamlit run mlunsupervisedapp.py"*
- This will automatically take you to a **local url** where you can use and play with the app!
### 4️⃣ Another (perhaps easier) Way to Run the App!
You can simply find the deployed version of the app on Streamlit Cloud here! 
- [Click here to access the app!](https://ortiz-data-science-portfolio-mlunsupervisedapp.streamlit.app/)

## 🖥️  App Features 
### 📊 Datasets Available
- Option to Upload Your Own 
- Iris Dataset 🌱
- Palmer's Penguins Dataset 🐧
### 🎯 Machine Learning Models
- K-Means Clustering
- Hierarchical Clustering
- Principal Component Analysis (PCA)
### 🛠️ Key Parameters You Can Tune 
- Number of clusters (k)
- Custom selection on which features to include 
### 📈 Visuals and Outputs
- **Cluster Scatterplots** (2-dimensional PCA visualizations)
- **True Labels Comparison Plots**
- **Dendrogram** for Hierarchical Clustering
- **PCA** Biplots and Cumulative Explained Variance plots
- **Silhouette Score and Elbow Method Plots**

## 📚 References and Documentations
- [Streamlit API Cheat Sheet](https://docs.streamlit.io/develop/quick-reference/cheat-sheet)
- [Scikit Learn Clustering User Guide](https://scikit-learn.org/stable/modules/clustering.html)
- [Scipy K-Means Clustering Guide](https://docs.scipy.org/doc/scipy/reference/cluster.vq.html)
- [Scipy Hierarchical Clustering Guide](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
- [Scikit Learn PCA User Guide](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Seaborn API Reference Sheet](https://seaborn.pydata.org/api.html)

## 📸 Visual Examples
<img width="600" alt="K-means cluster" src="https://github.com/user-attachments/assets/f2811b56-744a-4cf5-83f3-c5c3bb5392de" />
<img width="600" alt="Dendrogram" src="https://github.com/user-attachments/assets/559333c1-16f4-4bc8-bdd0-8a3e6e1bf474" />
<img width="600" alt="PCA Biplot" src="https://github.com/user-attachments/assets/bed816e8-53f1-4b73-9d12-dd4698f17d72" />
<img width="600" alt="Combined variance plot" src="https://github.com/user-attachments/assets/989c2453-996d-4597-9de3-a294386602a0" />
