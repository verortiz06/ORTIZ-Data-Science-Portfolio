# Streamlit App: Supervised Machine Learning
## 📌 Project Overview 
This project is centered around creating an interactive Streamlit app that allows users to explore core concepts of classification models in machine learning! In this streamlit app, users can:
- Select from preloaded datasets (such at the Iris or Titanic datasets) or upload their own
- Customize the chosen dataset's features and let the imagination run wild while building hypothetical datasets
- Choose from popular classification models (Logistic Regression, Decision Tree, KNN)
- Experiment with hyperparameters and see how the model's performance changes
- Visualize the model's performance through confusion matrices and ROC curve visualizations
- From hypotheticals created with the features, inspect the model's predictions

## ⚙️ How to Run This Project
### 1️⃣ Ensure that you have imported Streamlit
### 2️⃣ Install Required Libraries and Packages
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the App
- Within your terminal, locate the correct location of the MLStreamlitApp file by using commands **"ls"** to explore your desktop and **"cd"** to "double-click" into your folders until you find *mlstreamlitapp.py*
- Still within your terminal, type *"streamlit run mlstreamlitapp.py"*
- This will automatically take you to a **local url** where you can use and play with the app!
### 4️⃣ Another Way to Run
You can find the deployed version of the app on Streamlit Cloud here! 
- [Click here to access the app!](https://ortiz-data-science-portfolio-mlapp.streamlit.app/)

## 🖥️  App Features 
### 📊 Datasets Available
- Option to Upload Your Own 
- Titanic Dataset 🚢
- Iris Dataset 🌱
### 🎯 Machine Learning Models
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
### 🛠️ Hyperparameters You Can Tune 
- **Logistic Regression**: `maximum iterations`
- **Decision Tree**:  `maximum depth`
- **K-Nearest Neighbors**:  `number of neighbors (k)`
### 📈 Visuals and Outputs
- **Confusion matrices** with labeled axes
- **Performance metrics**: Accuracy, Precision, and ROC Curve (for binary tasks)
- **Prediction tables** illustrating the model's estimated probabilities of classifying correctly

## 📚 References and Documentations
- [Streamlit API Cheat Sheet](https://docs.streamlit.io/develop/quick-reference/cheat-sheet)
- [SciKit Learn User Guide](https://scikit-learn.org/stable/supervised_learning.html)
- [Seaborn API](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
- [Matplotlib User Guide](https://matplotlib.org/stable/users/index.html)

## 📸 Visual Examples
<img width="600" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/b09fcb7e-43f5-46c5-8929-ac9a2bcdd609" />
<img width="600" alt="ROC curve" src="https://github.com/user-attachments/assets/9d18bb1c-e06a-4d62-910a-f21aded961da" />
<img width="600" alt="Survival Pred" src="https://github.com/user-attachments/assets/033d1c12-3013-4a93-bd59-164da52d9694" />











