import streamlit as st 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

# Set Page Configuration
st.set_page_config(page_title="ML Dataset Analysis", layout="wide")

st.title("Machine Learning Dataset Analysis and Visualization")

# Upload dataset
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=['csv'])
df = pd.DataFrame()

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Overview")
    st.dataframe(df.head())
else:
    st.warning("Please upload a dataset to begin analysis.")

if not df.empty:
    st.sidebar.subheader("Quick Data Info")
    st.sidebar.write(f"**Rows:** {df.shape[0]}")
    st.sidebar.write(f"**Columns:** {df.shape[1]}")
    
    # Column Selector
    selected_columns = st.sidebar.multiselect("Select Columns to Display", df.columns.tolist(), default=df.columns.tolist())
    st.subheader("Selected Data Preview")
    st.dataframe(df[selected_columns])

    # Handling Missing Values
    st.subheader("Handling Missing Values")
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    
    if not missing_values.empty:
        st.write("**Columns with Missing Values:**")
        st.dataframe(missing_values.to_frame(name="Missing Count"))

        missing_action = st.radio("Select an action to handle missing values:", 
                                  ("None", "Drop Rows", "Drop Columns", "Fill with Mean", "Fill with Median", "Fill with Mode"))
        
        if missing_action == "Drop Rows":
            df.dropna(inplace=True)
            st.success("Rows with missing values have been dropped.")
        elif missing_action == "Drop Columns":
            df.dropna(axis=1, inplace=True)
            st.success("Columns with missing values have been dropped.")
        elif missing_action == "Fill with Mean":
            df.fillna(df.mean(), inplace=True)
            st.success("Missing values have been filled with the column mean.")
        elif missing_action == "Fill with Median":
            df.fillna(df.median(), inplace=True)
            st.success("Missing values have been filled with the column median.")
        elif missing_action == "Fill with Mode":
            df.fillna(df.mode().iloc[0], inplace=True)
            st.success("Missing values have been filled with the column mode.")
    else:
        st.success("No missing values found in the dataset.")

    # Feature: Create New Column
    st.subheader("Create New Column Based on Existing Ones")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) >= 2:
        col1, col2 = st.selectbox("Select First Column", numeric_columns), st.selectbox("Select Second Column", numeric_columns)
        operation = st.selectbox("Select Operation", ["Addition", "Subtraction", "Multiplication", "Division"])
        new_column_name = st.text_input("Enter New Column Name", f"{col1}_{operation.lower()}_{col2}")

        if st.button("Create Column"):
            if new_column_name in df.columns:
                st.warning("A column with this name already exists. Choose a different name.")
            else:
                if operation == "Addition":
                    df[new_column_name] = df[col1] + df[col2]
                elif operation == "Subtraction":
                    df[new_column_name] = df[col1] - df[col2]
                elif operation == "Multiplication":
                    df[new_column_name] = df[col1] * df[col2]
                elif operation == "Division":
                    df[new_column_name] = df[col1] / df[col2].replace(0, np.nan)  # Avoid division by zero
                
                st.success(f"New column '{new_column_name}' added successfully!")
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                 
                st.dataframe(df.head())

    else:
        st.warning("At least two numeric columns are required to create a new column.")

    # Dataset Statistics
    st.subheader("Dataset Summary")
    st.write(df.describe().T)

    # Data Types
    st.subheader("Column Data Types")
    st.write(df.dtypes)

    # Visualization Options
    st.sidebar.subheader("Visualization Options")
    plot_type = st.sidebar.selectbox("Select Plot Type", ["Heatmap", "Pie Chart", "Value Counts", "Scatter Plot", "Pair Plot", "Custom Plot"])

    if plot_type == "Heatmap":
        st.subheader("Heatmap Visualization")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            fig, ax = plt.subplots()
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns available for heatmap visualization.")

    elif plot_type == "Pie Chart":
        st.subheader("Pie Chart Visualization")
        target_col = st.selectbox("Select Column for Pie Chart", df.columns.tolist())
        if st.button("Generate Pie Chart"):
            fig, ax = plt.subplots()
            df[target_col].value_counts().plot.pie(autopct="%1.1f%%", shadow=True, startangle=90, ax=ax)
            st.pyplot(fig)

    elif plot_type == "Value Counts":
        st.subheader("Value Counts Plot")
        primary_col = st.selectbox("Select Column", df.columns.tolist())
        if st.button("Generate Value Counts Plot"):
            fig, ax = plt.subplots()
            df[primary_col].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)

    elif plot_type == "Scatter Plot":
        st.subheader("Scatter Plot Visualization")
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_columns) >= 2:
            x_axis = st.selectbox("Select X-axis", numeric_columns)
            y_axis = st.selectbox("Select Y-axis", numeric_columns)
            if st.button("Generate Scatter Plot"):
                fig, ax = plt.subplots()
                sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)
                st.pyplot(fig)
        else:
            st.warning("At least two numeric columns are required for a scatter plot.")

    elif plot_type == "Pair Plot":
        st.subheader("Pair Plot Visualization")
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_pair_columns = st.multiselect("Select Columns for Pair Plot", numeric_columns, default=numeric_columns[:3])
        if st.button("Generate Pair Plot"):
            if len(selected_pair_columns) >= 2:
                fig = sns.pairplot(df[selected_pair_columns])
                st.pyplot(fig)
            else:
                st.warning("Please select at least two columns for a pair plot.")

    elif plot_type == "Custom Plot":
        st.subheader("Customizable Plot")
        custom_plot_type = st.selectbox("Select Type of Plot", ["line", "bar", "hist", "box", "kde"])
        custom_columns = st.multiselect("Select Columns", df.columns.tolist())
        if st.button("Generate Plot"):
            fig, ax = plt.subplots()
            df[custom_columns].plot(kind=custom_plot_type, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Please select at least one column.")
    
    st.sidebar.markdown("---")
    st.sidebar.info("A Simple EDA App for full analysis of ML Datasets. Upload a CSV file to explore its structure and visualizations.")



