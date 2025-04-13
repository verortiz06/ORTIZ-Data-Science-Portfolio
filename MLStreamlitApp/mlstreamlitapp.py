import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------------------------
# Application Information
# ----------------------------------------------
st.title("Supervised Machine Learning Playground! 🛝")
st.markdown("""
## About This Application:
This interactive application allows you to upload datasets, experiment with hyperparameters, and observe how you can affect 
the model's training and performance.
""")
st.info("Let's build a machine learning model!")
#
# -----------------------------------------------
# Uploading/Selecting a Dataset
# ----------------------------------------------
st.sidebar.header("Step 1: Upload or Select a Dataset")

# Putting in classic sample datasets
sample_datasets = {
    "Iris Dataset": sns.load_dataset("iris"),
    "Titanic Dataset": sns.load_dataset("titanic").dropna(subset=["age"])
}

# Adding in a select box where they can choose from the above datasets of upload their own
dataset_options = st.sidebar.selectbox("Choose a sample dataset or upload your own:", ["Upload Your Own"] + list(sample_datasets))

# If the user chooses to upload their own dataset
if dataset_options == "Upload Your Own":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file.")
else:
    df = sample_datasets[dataset_options]
    if dataset_options == "Titanic Dataset":
        df = df.dropna(subset = ['age'])
        df = pd.get_dummies(df, columns = ["sex"], drop_first = True)
        features = ["pclass", "age", "sibsp", "parch", "fare", "sex_male"]
        df = df[features + ["survived"]]  # Keep only selected features and target

# Showing a preview of the dataset before any machine learning
st.write("## Dataset Preview")
st.dataframe(df.head())

# -----------------------------------------------
# Choosing a Target Variable for the Dataset
# ----------------------------------------------
st.sidebar.header("Step 2: Choose Target Variable")

# Creating selectbox so the user can choose the target variable
target_col = st.sidebar.selectbox("Select the column to predict:", df.columns)

# Include all features instead of the target in the dataset
selected_features = [col for col in df.columns if col != target_col]

# Create feature and target sets based on selection
X = df[selected_features]
y = df[target_col]

# -----------------------------------------------
# Choosing a Machine Learning Model
# ----------------------------------------------
# Sidebar section to choose the ML model
st.sidebar.header("Step 3: Select Your Machine Learning Model")
model_choice = st.sidebar.selectbox("Choose a model:", ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors"])

# -----------------------------------------------
# Playing with Hyperparameters
# ----------------------------------------------
# Sidebar section to adjust hyperparameters based on model choice
st.sidebar.header("Step 4: Set Hyperparameters")
if model_choice == "Logistic Regression":
    max_iter = st.sidebar.slider("Maximum Iterations:", 100, 1000, 200, step=50)
    model = LogisticRegression(max_iter=max_iter)
elif model_choice == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth:", 1, 20, 5)
    model = DecisionTreeClassifier(max_depth=max_depth)
elif model_choice == "K-Nearest Neighbors":
    n_neighbors = st.sidebar.slider("Number of Neighbors (K):", 1, 20, 5)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

# -----------------------------------------------
# Customizing the Train-Test Split Ratio
# ----------------------------------------------
# Sidebar section to define the train-test split ratio
st.sidebar.header("5. Train-Test Split")
test_size = st.sidebar.slider("Test Set Size (%):", 10, 50, 20)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=test_size/100,
                                                    random_state=41)

# Training the model
model.fit(X_train, y_train)

# Using the data for predictions
y_pred = model.predict(X_test)

# Calculating performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)

# Display performance metrics
st.write("## Model Performance Metrics")
st.metric("Accuracy:", f"{accuracy:.3f}")
st.metric("Precision:", f"{precision:.3f}")
