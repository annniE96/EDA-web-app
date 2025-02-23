import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

def convert_to_numeric(value):
    if isinstance(value, str):
        # Remove non-numeric characters and convert to float
        numeric_value = ''.join(filter(str.isdigit, value))
        return float(numeric_value) if numeric_value else np.nan
    return value

st.set_page_config(page_title="ML Dataset Analysis", layout="wide")

st.title("Machine Learning Dataset Analysis and Visualization")

# Dataset uploading
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=['csv'])

if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Overview")
    st.dataframe(st.session_state.df.head())
else:
    st.warning("Please upload a dataset to begin analysis.")

if not st.session_state.df.empty:
    df = st.session_state.df  # Use this for brevity in the rest of the code
    
    st.sidebar.subheader("Quick Data Info")
    st.sidebar.write(f"**Rows:** {df.shape[0]}")
    st.sidebar.write(f"**Columns:** {df.shape[1]}")

    # Selection of columns
    selected_columns = st.sidebar.multiselect("Select Columns to Display", df.columns.tolist(), default=df.columns.tolist())
    st.subheader("Selected Data Preview")
    st.dataframe(df[selected_columns])

    # Code to handle missing values
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

    # Creating new columns with the help of existing ones
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
                    df[new_column_name] = df[col1] / df[col2]
                    df[new_column_name].replace([np.inf, -np.inf], np.nan, inplace=True) #handle inf values
                    df[new_column_name].fillna(0, inplace=True) #handle nan values
                st.success(f"New column '{new_column_name}' added successfully!")
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                st.dataframe(df.head())
    else:
        st.warning("At least two numeric columns are required to create a new column.")

    # Automated feature engineering
    st.subheader("Automated Feature Engineering")
    if st.button("Generate Polynomial Features"):
        original_columns = set(df.columns)
        poly = PolynomialFeatures(degree=2, interaction_only=True)
        poly_features = poly.fit_transform(df[numeric_columns])
        new_columns = []
        for col in poly.get_feature_names_out(numeric_columns):
            new_col = col
            i = 1
            while new_col in original_columns or new_col in new_columns:
                new_col = f"{col}_{i}"
                i += 1
            new_columns.append(new_col)
        poly_features_df = pd.DataFrame(poly_features, columns=new_columns)
        df = pd.concat([df, poly_features_df], axis=1)
        st.success("Polynomial features generated successfully!")
        st.dataframe(df.head())

    # Anomaly detection algorithms
    st.subheader("Anomaly Detection")
    if st.button("Detect Anomalies"):
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        df['anomaly'] = iso_forest.fit_predict(df[numeric_columns])
        st.success("Anomalies detected!")
        st.write("Anomalies found:", (df['anomaly'] == -1).sum())
        fig, ax = plt.subplots()
        if len(numeric_columns) >= 2:
             sns.scatterplot(data=df, x=numeric_columns[0], y=numeric_columns[1], hue='anomaly', ax=ax)
        else:
            st.warning("Need at least two numeric columns for scatter plot of anomalies")

        st.pyplot(fig)

    # Time series analysis tools
    st.subheader("Time Series Analysis")
    date_columns = df.select_dtypes(include=['datetime64[ns]']).columns
    if len(date_columns) > 0:
        selected_date_col = st.selectbox("Select date column for time series analysis", date_columns)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist() # Get updated numeric columns
        selected_value_col = st.selectbox("Select value column for time series analysis", numeric_columns)
        if st.button("Perform Time Series Decomposition"):

            df[selected_date_col] = pd.to_datetime(df[selected_date_col])
            df = df.set_index(selected_date_col)
            try:
                result = seasonal_decompose(df[selected_value_col], model='additive', period=30)
                fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12))
                result.observed.plot(ax=ax1)
                result.trend.plot(ax=ax2)
                result.seasonal.plot(ax=ax3)
                result.resid.plot(ax=ax4)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Time series decomposition failed: {e}")
    else:
        st.warning("No datetime columns found for time series analysis.")

    # Advanced statistical tests
    st.subheader("Advanced Statistical Tests")
    test_type = st.selectbox("Select statistical test", ["Chi-square Test", "T-Test", "ANOVA"])

    test_explanations = {  # Define the dictionary here, inside the if statement.
        "Chi-square Test": """
            The Chi-square test assesses the independence of two categorical variables. It calculates a statistic (χ²) comparing observed to expected frequencies under the assumption of independence.

            *Chi-square statistic (χ²):*  Measures the discrepancy between observed and expected frequencies. Larger values indicate a greater difference.
            *p-value:*  Represents the probability of observing a test statistic as extreme as, or more extreme than, the statistic obtained, assuming that the null hypothesis (independence) is true.  A small p-value (typically ≤ 0.05) suggests that the variables are likely dependent.
            """,
        "T-Test": """
            The T-test assesses if the means of two groups are significantly different.

            *T-statistic:* A ratio of the difference between the group means and the standard error of the difference.
            *p-value:*  The probability of observing a t-statistic as extreme as, or more extreme than, the statistic obtained, assuming that the null hypothesis (no difference in means) is true.  A small p-value (typically ≤ 0.05) suggests a significant difference in the means.
            """,
        "ANOVA": """
            ANOVA (Analysis of Variance) tests if the means of two or more groups are significantly different.

            *F-value:*  The ratio of between-group variance to within-group variance.  A larger F-value suggests greater differences between group means.
            *p-value:*  The probability of observing an F-statistic as extreme as, or more extreme than, the statistic obtained, assuming that the null hypothesis (all group means are equal) is true. A small p-value (typically ≤ 0.05) suggests that at least one group mean is significantly different.
            """
    }

    # Display the explanation for the selected test
    st.info(test_explanations[test_type])

    if test_type == "Chi-square Test":
        cat_columns = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_columns) >= 2:
            col1, col2 = st.selectbox("Select first categorical column", cat_columns), st.selectbox("Select second categorical column", cat_columns)
            if st.button("Perform Chi-square Test"):
                contingency_table = pd.crosstab(df[col1], df[col2])
                try:
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                    st.write(f"Chi-square statistic: {chi2}")
                    st.write(f"p-value: {p_value}")
                    st.info(f"Interpretation: A p-value <= 0.05 suggests a significant association between {col1} and {col2}.")


                except ValueError as e:
                    st.error(f"Chi-square test failed: {e}.  Ensure expected frequencies are not too low.")


        else:
            st.warning("At least two categorical columns are required for Chi-square test.")

    elif test_type == "T-Test":
        if len(numeric_columns) >= 1:
            col = st.selectbox("Select numeric column for T-Test", numeric_columns)
            group_col = st.selectbox("Select grouping column", df.columns)
            if st.button("Perform T-Test"):
                try:
                    group1 = df[df[group_col] == df[group_col].unique()[0]][col]
                    group2 = df[df[group_col] == df[group_col].unique()[1]][col]
                    t_stat, p_value = stats.ttest_ind(group1, group2)
                    st.write(f"T-statistic: {t_stat}")
                    st.write(f"p-value: {p_value}")
                    st.info(f"Interpretation: A p-value <= 0.05 suggests a significant difference in means between the groups defined by {group_col}.")
                except Exception as e:
                    st.error(f"T-test failed: {e}.  Ensure grouping column has at least two unique values and there are no NaN values in the selected columns.")
        else:
            st.warning("At least one numeric column is required for T-Test.")

    elif test_type == "ANOVA":
        if len(numeric_columns) >= 1:
            col = st.selectbox("Select numeric column for ANOVA", numeric_columns)
            group_col = st.selectbox("Select grouping column", df.columns)
            if st.button("Perform ANOVA"):
                try:
                    groups = [group[col].values for name, group in df.groupby(group_col) if len(group) > 0]  # Use .values to avoid pandas errors

                    f_value, p_value = stats.f_oneway(*groups)
                    st.write(f"F-value: {f_value}")
                    st.write(f"p-value: {p_value}")
                    st.info(f"Interpretation: A p-value <= 0.05 suggests there's a significant difference between the means of at least one group defined by {group_col}.")

                except Exception as e:
                    st.error(f"ANOVA failed: {e}.  Ensure grouping column has at least two groups and no NaN values.")
        else:
            st.warning("At least one numeric column is required for ANOVA.")



    # Dataset Statistics using describe
    st.subheader("Dataset Summary")
    st.write(df.describe().T)

    # Data Types
    st.subheader("Column Data Types")
    st.write(df.dtypes)

    # Visualization Options
    st.sidebar.subheader("Visualization Options")
    plot_type = st.sidebar.selectbox("Select Plot Type", ["Heatmap", "Pie Chart", "Value Counts", "Scatter Plot", "Pair Plot", "Custom Plot"])

    # Get the latest numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if plot_type == "Heatmap":
        st.subheader("Heatmap Visualization")
        if numeric_columns:
            fig, ax = plt.subplots()
            sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", ax=ax)
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
        selected_pair_columns = st.multiselect("Select Columns for Pair Plot", numeric_columns, default=numeric_columns[:min(3, len(numeric_columns))])
        if st.button("Generate Pair Plot") and len(selected_pair_columns) > 1:
            fig = sns.pairplot(df[selected_pair_columns])
            st.pyplot(fig)
        elif len(selected_pair_columns) <= 1:
            st.warning("Please select at least two columns for the pair plot.")

    elif plot_type == "Custom Plot":
        st.subheader("Custom Plot")
        st.write("You can add your custom plot logic here.")

    # Update the session state DataFrame
    st.session_state.df = df



