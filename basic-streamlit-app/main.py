#Importing pandas and streamlit
import streamlit as st
import pandas as pd

#Creating the title and the short description of the app
st.title("Looking at Palmer's Penguins!")
st.markdown("In this dataset, you will see information regarding 343 penguins! The included data on these penguins includes the penguins' ID number, species, island, and sex. Additionally, there is data outlining the penguins' bill lengths and depths, flipper length, and body mass.")

#Adding a fun image of a penguin!
st.image("https://www.cabq.gov/artsculture/biopark/news/10-cool-facts-about-penguins/@@images/1a36b305-412d-405e-a38b-0947ce6709ba.jpeg")

#Importing the Palmer's Penguins dataset and also making sure that it's in the "data" folder of the repository
df = pd.read_csv("Data/penguins.csv")

#Adding these dashes creates a break between the filters I'm making
st.markdown("-----")

#This filer lets the user to select a species to look at from the dataset
st.header("Sort the penguins by species!") 
species = st.selectbox("Select species:", options = df["species"].unique()) #Adding the "unique" part is important because it makes sure that each species appears only once on the dropdown
species_filtered_df = df[df["species"] == species] #Creating a filtered dataframe with the species the user chooses
st.write(f"Penguins in species {species}:") 
st.dataframe(species_filtered_df) #Displays the filtered dataset by species that is chosen

#This is creating another filter almost identical to the one above, except it is for the penguins' island origins
st.markdown("-----")
st.header("Sort the penguins by island!")
island = st.selectbox("Select island:", options = df["island"].unique())
island_filtered_df = df[df["island"] == island]
st.write(f"Penguins in island {island}:")
st.dataframe(island_filtered_df)

#This filer allows the user to play around with a slider to filter through the flipper lengths 
st.markdown("-----")
st.header("Filter the penguins through flipper length!")
flipper_length = st.slider("Slide me to explore!", #Makes the slider and the description of it
                           min_value = df["flipper_length_mm"].min(), #Finds and assigns the minimum value from the dataset to the lower bound of the slider
                           max_value = df["flipper_length_mm"].max()) #Finds and assigns the maximum value from the dataset to the upper bound of the slider
st.write(f"Penguins with flipper lengths under {flipper_length}:") #Description of the filtered dataset that is spit out
st.dataframe(df[df["flipper_length_mm"] <= flipper_length]) #Displays the filtered dataset that the user chooses

st.markdown("-----")
st.markdown("Thank you for visiting, see you again soon! 👋")
st.image("https://i.pinimg.com/736x/69/c1/a2/69c1a2274cf1023501dbb82751cb26b9.jpg")