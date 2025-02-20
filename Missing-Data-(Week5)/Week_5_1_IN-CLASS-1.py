import pandas as pd            # Library for data manipulation
import seaborn as sns          # Library for statistical plotting
import matplotlib.pyplot as plt  # For creating custom plots
import streamlit as st         # Framework for building interactive web apps

# ================================================================================
#Missing Data & Data Quality Checks
#
# This lecture covers:
# - Data Validation: Checking data types, missing values, and ensuring consistency.
# - Missing Data Handling: Options to drop or impute missing data.
# - Visualization: Using heatmaps and histograms to explore data distribution.
# ================================================================================
st.title("Missing Data & Data Quality Checks")
st.markdown("""
This lecture covers:
- **Data Validation:** Checking data types, missing values, and basic consistency.
- **Missing Data Handling:** Options to drop or impute missing data.
- **Visualization:** Using heatmaps and histograms to understand data distribution.
""")

# ------------------------------------------------------------------------------
# Load the Dataset
# ------------------------------------------------------------------------------
# Read the Titanic dataset from a CSV file.
df = pd.read_csv("titanic.csv")

# ------------------------------------------------------------------------------
# Display Summary Statistics
# ------------------------------------------------------------------------------
# Show key statistical measures like mean, standard deviation, etc.
st.write("**Summary Statistics**")
st.write(df.shape)
st.dataframe(df.describe()) # describe looks at numeric columns only and gives summary statistics
# observations we can make based off of .describe()
# - we can see that we're missing a lot of age
# - something's happening with the fare and the minimum being zero
# - the minimum age being 18 and mean being 33 is weird; there were definitely minors on board
# - the missing values might be the kids on board
# the type of missing data we have: MNAR (intentional reason for not recording this obervation)

# ------------------------------------------------------------------------------
# Check for Missing Values
# ------------------------------------------------------------------------------
# Display the count of missing values for each column.
st.write("**Number of Missing Values by Column**")
# sum of missing values in a column
st.dataframe(df.isnull()) # casts a boolean mask on the entire dataframe; "does this observation have a value? True/False"
# if there is a checked box, there is a missing value
# False - there is something there (0); True - there is a missing value (1)
st.dataframe(df.isnull().sum()) # how many nulls we have in a column

# ------------------------------------------------------------------------------
# Visualize Missing Data
# ------------------------------------------------------------------------------
# Create a heatmap to visually indicate where missing values occur.
st.write("Heatmap of Missing Values")
fig, ax = plt.subplots() # from matlib pyplot; like putting a blank canvas up for the upcoming code
# figure, axes
sns.heatmap(df.isnull(), cmap = "viridis", cbar = False) # paint on the canvas
st.pyplot(fig) # shows the art/missing values

# ================================================================================
# Interactive Missing Data Handling
#
# Users can select a numeric column and choose a method to address missing values.
# Options include:
# - Keeping the data unchanged
# - Dropping rows with missing values
# - Dropping columns if more than 50% of the values are missing
# - Imputing missing values with mean, median, or zero
# ================================================================================
st.subheader("Handle Missing Data")

column = st.selectbox("Choose a column to fill", df.select_dtypes(include=["number"]).columns) # grabs only numberic columns

# Apply the selected method to handle missing data.
#st.dataframe(df[column]) # only access that column, not the whole dataframe

# also want to provide a method for imputing/handling the missing data in those columns
method = st.radio("Choose a method:", 
         ["Original DF", "Drop Rows", "Drop Columns (>50% missing)", 
        "Impute Mean", "Impute Median", "Impute Zero"]) 
# radio box lets you see all options at once for deleting or imputing data, including the original dataframe

# Work on a copy of the DataFrame so the original data remains unchanged.
# df is going to remain untouched
# df_clean is going to be our imputation/deletion dataframe
df_clean = df.copy() # YOU NEED TO DO THE .COPY() or else the original df WILL change

if method == "Original DF":
    pass # can use this keyword to skip through the entire conditional statement
elif method == "Drop Rows":
    df_clean = df_clean.dropna(subset = [column]) # function that drops all rows that have a missing value
    # adding subset = [column] just drops null values from selected column
elif method == "Drop Columns (<50% Missing)":
    # Drop columns where more than 50% of the values are missing.
    df_clean == df_clean.drop(columns = df_clean.columns[df_clean.isnull().mean() > 0.50])
elif method == "Impute Mean":
    # replace values in selected column with the column's mean
    df_clean[column] = df_clean[column].fillna(df[column].mean())
elif method == "Impute Median":
    # replace values in selected column with the column's median
    df_clean[column] = df_clean[column].fillna(df[column].median())
elif method == "Impute Zero": 
    #replace missing values with zero - pretty common
    df_clean[column] = df_clean[column].fillna(0)

# you don't need an else statement because the radio button shows us all the options we have

st.subheader("Cleaned Data Distribution")
fig, ax = plt.subplots() #creating blank canvas
sns.histplot(df_clean[column], kde = True) #histogram showing distribution; painted on canvas
st.pyplot(fig) #revealing the painting
# st.dataframe(df_clean)
st.write(df_clean.describe()) # shows us summary stats for new dataframe
# age column had so many missing values, imputing anything skews the entire distribution
# we have to be careful when imputing data
# age was a variable where all minors were unaccounted for -- this was MNAR (missing not at random)
# imputation methods (especially for MNAR) are very risky

# ------------------------------------------------------------------------------
# Compare Data Distributions: Original vs. Cleaned
#
# Display side-by-side histograms and statistical summaries for the selected column.
# ------------------------------------------------------------------------------

