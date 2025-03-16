# TidyData Project
## 📌 Overview
This project is centered around demonstrating the process of cleaning data and restructuring it into a **tidy data** format. The main characteristics that define tidy datasets are:
- Each **variable** having its own column
- Each **observation** forming its own row
- Each type of **observational unit** forms its own table

The main purpose of transforming messy datasets into tidy ones is to facilitate analysis, calculations, creating visualizations, and machine learning.

## ⚙️ How to Run This Project
### 1️⃣ Install Necessary Libraries
First ensure that you have Python installed, and then install the following libraries:
```bash
pip install pandas seaborn matplotlib.pyplot
```
### 2️⃣ Run the Notebook
Open the Jupyter Notebook or use the Python code to see the data cleaning and tidy data process. Make sure to run each cell in order to transform the data correctly!
### 3️⃣ What You Can Expect to See
The output should be:
- A cleaned and organized dataset in a "long" format instead of the originally "wide" format
- Correctly named and structured columns
- Properly formatted values and Member names
- Visualizations reflecting trends in the data

## 📊 Dataset Description
- Source: The dataset used in this project is the `mutant_moneyball.csv` which can be found within this repository
- Pre-processing steps used:
  1. **Converted the dataset from "wide" to "long" format** using  `melt()`
  2. **Extracted multiple variables from compounded column names** using `split()`, `replace()`, and `drop()`
  3. **Converted strings to numerical values** using `astype()`, `replace()`, and `to_numeric()` to remove dollar signs and commas from *Value* column
  4. **Correctly formatted categorical data** to fix *Member* names and make them readable
 
## 📑 References Used
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Tidy Data Principles, Hadley Wickham](https://vita.had.co.nz/papers/tidy-data.pdf)
- [Pandas API References](https://pandas.pydata.org/docs/reference/general_functions.html)

## 📸 Visual Examples
<img width="600" alt="Visualization_1" src="https://github.com/user-attachments/assets/cfba349f-92fc-4004-81dd-82d3ed8ba1a9" />
<img width="600" alt="Visualization_2" src="https://github.com/user-attachments/assets/66b5188c-0925-42e2-bf70-16ac085fa6c3" />



  
