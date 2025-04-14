# -----------------------------------------------
# Importing All Necessary Libraries
# ----------------------------------------------
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
import graphviz
from sklearn import tree

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

# -----------------------------------------------
# Step 1: Uploading/Selecting a Dataset
# ----------------------------------------------
st.sidebar.header("Step 1: Upload or Select a Dataset") # Creating a sidebar header for this step

# Putting in classic sample datasets
sample_datasets = {
    "Iris Dataset": sns.load_dataset("iris"),
    "Titanic Dataset": sns.load_dataset("titanic").dropna(subset=["age"])
}

# Adding in a select box where they can choose from the above datasets of upload their own
dataset_options = st.sidebar.selectbox("Choose a sample dataset or upload your own:", ["Upload Your Own"] + list(sample_datasets))

# If the user chooses to upload their own dataset
if dataset_options == "Upload Your Own":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"]) # Allows the user to upload their own dataset
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file. Error will occur and no Dataset Preview will be available due to there" \
        " not being a dataset uploaded yet!") # Will give a warning to the user that they need to upload something or choose a sample dataset

# If the user chooses to use one of the sample datasets (Titanic or Iris)
else: 
    df = sample_datasets[dataset_options]
    if dataset_options == "Titanic Dataset": # If the user chooses the Titanic Sample Dataset 
        df = pd.get_dummies(df, columns = ["sex"], drop_first = True) # Make sex a dummy variable
        features = ["pclass", "age", "sibsp", "parch", "fare", "sex_male"] # The features from the Titanic Dataset that will be included
        df = df[features + ["survived"]]  # Keep only selected features and target
        df["survived"] = df["survived"].astype("category") # This makes the "survived" variable a categorical one, which is necessary for the machine learning models we're using

        # Creating a section where the user can customize and play with the features of the Titanic Dataset
        st.sidebar.write("Build your own Titanic Passenger!") 
        custom_input = {
            "pclass": st.sidebar.slider("Passenger Class", int(df["pclass"].min()), int(df["pclass"].max())),
            "age": st.sidebar.slider("Age", int(df["age"].min()), int(df["age"].max())),
            "sibsp": st.sidebar.slider("Siblings/Spouses", int(df["sibsp"].min()), int(df["sibsp"].max())),
            "parch": st.sidebar.slider("Parents/Children", int(df["parch"].min()), int(df["parch"].max())),
            "fare": st.sidebar.slider("Fare Paid", int(df["fare"].min()), int(df["fare"].max())),
            "sex_male": st.sidebar.slider("Female (0) or Male (1)", int(df["sex_male"].min()), int(df["sex_male"].max()))
        }
    if dataset_options == "Iris Dataset": # If the user chooses the Iris Sample Dataset
        
        # Creating a section where the user can customize and play with the features of the Iris Dataset
        st.sidebar.write("Customize Input Features:")
        custom_input = {
            "sepal_length": st.sidebar.slider("Sepal Length", float(df["sepal_length"].min()), float(df["sepal_length"].max()), float(df["sepal_length"].mean())),
            "sepal_width": st.sidebar.slider("Sepal Width", float(df["sepal_width"].min()), float(df["sepal_width"].max()), float(df["sepal_width"].mean())),
            "petal_length": st.sidebar.slider("Petal Length", float(df["petal_length"].min()), float(df["petal_length"].max()), float(df["petal_length"].mean())),
            "petal_width": st.sidebar.slider("Petal Width", float(df["petal_width"].min()), float(df["petal_width"].max()), float(df["petal_width"].mean())),
    }
    
st.divider()

# Will show the user a preview of the dataset before any machine learning, customizations, or other changes
st.write("## 🔍 Dataset Preview")
st.markdown("Here you can see a general idea of what your chosen dataset" \
" looks like!")
st.dataframe(df.head()) # Displays only the first five rows of the dataset

st.divider()

# -----------------------------------------------
# Step 2: Choosing a Target Variable for the Dataset
# ----------------------------------------------
st.sidebar.header("Step 2: Choose Target Variable")
categorical_columns = df.select_dtypes(include=['category', 'object']).columns.tolist() # These lines of code will make it so that the user can only choose categorical variables as the target
target_col = st.sidebar.selectbox("Select a categorical column to predict:", categorical_columns)

# Define features (X) and target (y)
selected_features = [col for col in df.columns if col != target_col]
X = df[selected_features]
y = df[target_col]

# -----------------------------------------------
# Step 3: Choosing a Machine Learning Model
# ----------------------------------------------
# Sidebar section to choose the ML model
st.sidebar.header("Step 3: Select Your Machine Learning Model")
model_choice = st.sidebar.selectbox("Choose a model:", ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors"]) # The ML models we have as options
if model_choice == "Logistic Regression": # If the user chooses Logistic Regression, this explanation will appear
    st.sidebar.markdown("""
    **Logistic Regression** is a linear model used for binary or multiclass classification.

    It estimates the probability of a class using a logistic function.
    
    This machine learning model is great for understanding the impact of individual features
    on a specific outcome.
    """)
elif model_choice == "Decision Tree": # If the user chooses the Decision Tree, this explanation will appear
    st.sidebar.markdown("""
    **Decision Trees** are flowchart-like structures that make decusions by asking a series of yes/no questions.
    
    A decision tree is composed of nodes and branches that mimic human decision making!

    """)
elif model_choice == "K-Nearest Neighbors": # If the user chooses K-Nearest Neighbors, this explanation will appear
    st.sidebar.markdown("""
    **K-Nearest Neighbors (KNN)** is a simple, instance-based algorithm best suited for classification tasks 
    where outcomes are categorical.
    
    It classifies new samples based on the majority class of its **k** nearest data points. 
    
    Therefore, the model relies on the idea that similar data points (in the feature space) tend to have similar outcomes!
    
    """)

# -----------------------------------------------
# Step 4: Playing with Hyperparameters
# ----------------------------------------------
# Sidebar section to adjust hyperparameters based on the model choice
st.sidebar.header("Step 4: Set Hyperparameters")
if model_choice == "Logistic Regression": # If they choose a Logistic Regression model
    max_iter = st.sidebar.slider("Maximum Iterations:", 100, 1000, 200, step = 50) # This is the hyperparameter they can tune
    model = LogisticRegression(max_iter = max_iter)
elif model_choice == "Decision Tree": # If they choose a Decision Tree model
    max_depth = st.sidebar.slider("Max Depth:", 1, 20, 5) # This is the hyperparameter they can tune
    model = DecisionTreeClassifier(max_depth = max_depth)
elif model_choice == "K-Nearest Neighbors": # If they choose a KNN model
    n_neighbors = st.sidebar.slider("Number of Neighbors (K):", 1, 20, 5) # This is the hyperparameter they can tune
    model = KNeighborsClassifier(n_neighbors = n_neighbors)

# -----------------------------------------------
# Step 5: Customizing the Train-Test Split Ratio
# ----------------------------------------------
# Sidebar section to define the train-test split ratio
st.sidebar.header("Step 5: Define your Train-Test Split ratio")
test_size = st.sidebar.slider("Test Set Size (%):", 10, 50, 20)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=test_size/100,
                                                    random_state=41)

# -----------------------------------------------
# Making the model
# ----------------------------------------------
# Training the model
model.fit(X_train, y_train)

# Using the data for predictions
y_pred = model.predict(X_test)

# -----------------------------------------------
# Showing the Model's Performance
# ----------------------------------------------
# Calculating performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)

# Display performance metrics
st.write("## ⚙️ Model Performance Metrics")

# Accuracy
st.metric("Accuracy:", f"{accuracy:.3f}")
st.markdown("The accuracy metric you see here describes the model's overal percentage" \
"of correct classifications. The formula used to derive this number is: " \
" (True Positives + True Negatives) divided by (Total Positives + Total Negatives)")

# Precision
st.metric("Precision:", f"{precision:.3f}")
st.markdown("Precision is defined as the model's positive predictive" \
"value. In other words, of all predicted positives, how many were actually" \
"positive? The way this metric is predicted is True Positives divided by Total Positives")

# Confusion Matrix for Titanic Dataset
if dataset_options == "Titanic Dataset":
    st.subheader("Model Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot = True, cmap = 'Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt)
    st.markdown("""
A **confusion matrix** gives you a visual summary of how well the model is performing by comparing actual versus predicted.

- The **rows** represent the actual classes.
- The **columns** represent the predicted classes.

In a **binary classification** (such as the Titanic dataset predicting survival):
- **Top-left** = True Negatives (correctly predicted non-survivals)
- **Top-right** = False Positives (predicted survival, but actually didn’t)
- **Bottom-left** = False Negatives (predicted non-survival, but actually survived)
- **Bottom-right** = True Positives (correctly predicted survivals)
""") # Explanation the user will get for the Titanic Confusion Matrix

# Confusion Matrix explanation for Iris Dataset
if dataset_options == "Iris Dataset":
    st.subheader("Model Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    species_labels = model.classes_
    sns.heatmap(cm, annot = True, cmap = 'Blues',
                xticklabels = species_labels, 
                yticklabels = species_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt)
    st.markdown("""In a **multiclass classification** (such as this one!) which has three classes (due to there
                 being three species of iris), it shows how often the model confuses one class for another. The darker
                the square when the 'actual' data matches the 'predicted' data is, the better your model is!
""") # Explanation the user will get for the Iris Confusion Matrix

st.divider()

# Showing predicted vs actual values so they user can inspect
st.write("## 🕵️ Model Predictions")
st.markdown("The table below allows you to inspect the dataset by displaying the actual data versus" \
" the model's predicted values in the last two columns. With this, you can get another sense of which certain cases the model gets" \
" correct versus makes mistakes on.")
pred_df = X_test.copy()
pred_df['Actual'] = y_test
pred_df['Predicted'] = y_pred
st.dataframe(pred_df)

# The ROC Curve -- Will only appear for the Titanic Dataset since it is a binary
if len(model.classes_) == 2:  # ROC curves only works for binary classification
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    st.subheader("📈 ROC Curve")
    st.markdown("A ROC curve plots the True Positive Rate (TPR) against the False" \
    " Positive Rate (FPR). The ROC curve only works for binary classification models" \
    " and essentially, it illustrates how well the model is performing in discriminating" \
    " between two classes compared to just a random 50-50 guess! Therefore, the more concave" \
    " the ROC curve is in comparison to the 'Random Guess' line, the better your model is.") # Explanation for the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label = "Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc = "lower right")
    st.pyplot(plt)


# -----------------------------------------------
# Displaying Predictions
# ----------------------------------------------

# Model Predictions if the user chooses the Iris Dataset
if dataset_options == "Iris Dataset":
    custom_df = pd.DataFrame([custom_input])
    proba = model.predict_proba(custom_df)[0]
    df_prediction_proba = model.predict_proba(custom_df)[0]
    
    st.subheader("🪴 Predicted Species")
    st.write("If you play around with the customizable input features in the sidebar, " \
    "you will be able to see the model's predictions below! The percentages illustrate the model's" \
    " estimated probability of guessing the species correct based on your custom input.")
    st.dataframe(
       pd.DataFrame([df_prediction_proba], columns = ["Virginica", "Versicolor", "Setosa"]),
       column_config = {
           "Virginica": st.column_config.ProgressColumn("Virginica", format = '%.2f', min_value = 0, max_value = 1),
           "Versicolor": st.column_config.ProgressColumn("Versicolor", format = '%.2f', min_value = 0, max_value = 1),
           "Setosa": st.column_config.ProgressColumn("Setosa", format = '%.2f', min_value = 0, max_value = 1)
        },
        hide_index = True
    )

# Model Predictions if the user chooses the Titanic Dataset
if dataset_options == "Titanic Dataset":
    custom_df = pd.DataFrame([custom_input])
    proba = model.predict_proba(custom_df)[0]
    df_prediction_proba = model.predict_proba(custom_df)[0]
    
    st.subheader("Survival Prediction 🚢🧊")
    st.write("If you play around with the customizable input features in the sidebar, " \
    "you will be able to see the model's predictions below! The percentages illustrate the model's" \
    " estimated probability of predicting correctly based on your custom input.")
    st.dataframe(
        pd.DataFrame([df_prediction_proba], columns = ["Did Not Survive (0)", "Survived (1)"]),
        column_config = {
            "Did Not Survive (0)": st.column_config.ProgressColumn("Did Not Survive (0)", format = '%.2f', min_value = 0, max_value = 1),
            "Survived (1)": st.column_config.ProgressColumn("Survived (1)", format = '%.2f', min_value = 0, max_value = 1)
        },
        hide_index = True
    )

