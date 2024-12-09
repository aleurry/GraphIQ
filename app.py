import streamlit as st
import pandas as pd
import plotly.express as px
import io
import numpy as np
from scipy import stats

import pandas as pd
import numpy as np

def clean_data(df, missing_threshold=0.5, zscore_threshold=3, numeric_range=None):
    
    # Step 1: Drop columns or rows with excessive missing values
    missing_col_percent = df.isnull().mean()  # Percentage of missing values per column
    df = df.loc[:, missing_col_percent < missing_threshold]
    missing_row_percent = df.isnull().mean(axis=1)  # Percentage of missing values per row
    df = df.loc[missing_row_percent < missing_threshold]
    if "Name" in df.columns:
        df = df[df["Name"].notna()]  # Remove rows where 'Name' is NaN
        df = df[df["Name"] != ""]  # Remove rows where 'Name' is an empty string

    # Step 2: Handling Text in Number Columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in df.columns.difference(numeric_cols):
        # Check if majority of values are numeric
        if df[col].astype(str).str.isnumeric().mean() > 0.5:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Step 3: Handling missing values
    for col in df.select_dtypes(include=["number"]).columns:
        mean_value = df[col].mean()  # Calculate the mean
        rounded_mean = np.ceil(mean_value)  # Round up the mean
        df[col].fillna(rounded_mean, inplace=True)  # Fill missing values with the rounded-up mean
    # Process "Name" and "name" columns
    for col in ["Name", "name"]:
        if col in df.columns:
            # Replace missing or empty names with "Unknown"
            df[col] = df[col].fillna("Unknown")
            df[col] = df[col].replace("", "Unknown")

            # Replace names containing numbers with "Unknown"
            df[col] = df[col].apply(lambda x: "Unknown" if any(char.isdigit() for char in x) else x)


    # Example: Fill missing values for other categorical columns with mode
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col != "Name":  # Exclude the 'Name' column
            mode_value = df[col].mode()[0]  # Get the mode of the column
            df[col].fillna(mode_value, inplace=True)  # Fill missing values with the mode

    # Step 4: Outlier Detection and Handling
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col]))
        df[col] = np.where(
            z_scores > zscore_threshold,
            np.where(df[col] > df[col].mean(), df[col].quantile(0.95), df[col].quantile(0.05)),
            df[col]
        )

    # Step 5: Handle out-of-range values
    if numeric_range:
        for col, (min_val, max_val) in numeric_range.items():
            if col in df.columns:
                df[col] = np.clip(df[col], min_val, max_val)

    # Step 6: Handle duplicate rows
    df = df.drop_duplicates()

    return df


# Main Streamlit app
def main():
    st.title("CSV Data Cleaning and Visualization")

    # Step 1: File Upload
    uploaded_file = st.file_uploader("Upload a CSV File", type="csv")
    
    if uploaded_file is not None:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded CSV:")
        st.dataframe(df)

        # Step 2: Data Cleaning Option
        if st.checkbox("Clean Data"):
            df = clean_data(df)
            st.write("Cleaned Data:")
            st.dataframe(df)

        # Step 3: Visualization Options
        if st.checkbox("Visualize Data"):
            st.subheader("Choose Visualization Type")
            visualization_type = st.selectbox(
                "Select a chart type:",
                ["Line Graph", "Column Graph", "Heatmap", "Radial Chart", "Funnel Chart"]
            )

            # Allow user to select columns for visualization
            st.subheader("Select Columns for Visualization")
            numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
            if len(numeric_columns) < 1:
                st.warning("No numeric columns available for visualization.")
                return

            x_axis = st.selectbox("X-Axis", options=numeric_columns)
            y_axis = st.selectbox("Y-Axis", options=numeric_columns)

            # Generate the selected visualization
            fig = None
            if visualization_type == "Line Graph":
                fig = px.line(df, x=x_axis, y=y_axis, title="Line Graph")
            elif visualization_type == "Column Graph":
                fig = px.bar(df, x=x_axis, y=y_axis, title="Column Graph")
            elif visualization_type == "Heatmap":
                fig = px.imshow(df.corr(), title="Heatmap")
            elif visualization_type == "Radial Chart":
                fig = px.line_polar(df, r=y_axis, theta=x_axis, title="Radial Chart")
            elif visualization_type == "Funnel Chart":
                fig = px.funnel(df, x=x_axis, y=y_axis, title="Funnel Chart")

            if fig:
                st.plotly_chart(fig)

        # Step 4: Report Download
        if st.checkbox("Download Cleaned Data as Report"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Cleaned Data")
            output.seek(0)
            st.download_button(
                label="Download Excel Report",
                data=output,
                file_name="cleaned_data_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
