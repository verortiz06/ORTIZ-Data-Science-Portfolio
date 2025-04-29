# -----------------------------------------------
# Importing All Necessary Libraries
# ----------------------------------------------
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
st.title("Supervised Machine Learning Playground! 🛝") # Creating a title for the app
st.markdown(""" 
## 📋 About This Application:
This interactive application allows you to upload datasets, experiment with hyperparameters, and observe how you can affect 
the model's training and performance.
""") # App description and explanation
st.info("Let's build a machine learning model!")
