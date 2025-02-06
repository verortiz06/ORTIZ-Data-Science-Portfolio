import streamlit as st
import pandas as pd

# ================================
# Step 1: Displaying a Simple DataFrame in Streamlit
# ================================

st.subheader("Now, let's look at some data!")

# Creating a simple DataFrame manually
# This helps students understand how to display tabular data in Streamlit.
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
})

# Displaying the table in Streamlit
# st.dataframe() makes it interactive (sortable, scrollable)
st.write("Here's a simple table:")
st.dataframe(df)

# ================================
# Step 2: Adding User Interaction with Widgets
# ================================

# Using a selectbox to allow users to filter data by city
# Students learn how to use widgets in Streamlit for interactivity
# SIMPLE QUERIES OVER A DATA FRAME

city = st.selectbox("Select a city", df["City"].unique())

# lets the users select a city from our dataframe
# first parameter is the text on top, second one is list of values
# could manually create a list, but that might take forever
# we can use pandas logic and methods instead to create the list for us
# access the city column
# use method unique
# you get a dropdown menu of all the values in the city column
# but, we want our buttons to actually do something; do something with the selectbox

# Filtering the DataFrame based on user selection

filtered_df = df[df["City"] == city]

# Display the filtered results
st.write(f"People in {city}:")
st.dataframe(filtered_df)

# ================================
# Step 3: Importing Data Using a Relative Path
# ================================

# Now, instead of creating a DataFrame manually, we load a CSV file
# This teaches students how to work with external data in Streamlit
# # Ensure the "data" folder exists with the CSV file
# Display the imported dataset

# can't just copy the file path as usual - copy the relative path for the file you want to reference
df2 = pd.read_csv("Data/sample_data.csv")
st.dataframe(df2)


# Using a selectbox to allow users to filter data by city
# Students learn how to use widgets in Streamlit for interactivity
salary = st.slider("Choose a salary range:",
                   min_value = df2["Salary"].min(),
                   max_value = df2["Salary"].max())

# Filtering the DataFrame based on user selection
st.write(f"Salaries under {salary}:")
st.dataframe(df2[df2["Salary"] <= salary])

# Display the filtered results

# ================================
# Summary of Learning Progression:
# 1️⃣ Displaying a basic DataFrame in Streamlit.
# 2️⃣ Adding user interaction with selectbox widgets.
# 3️⃣ Importing real-world datasets using a relative path.
# ================================