import streamlit as st
import pandas as pd
import plotly.express as px
import io
import numpy as np
from scipy import stats
from io import BytesIO
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows

# Data cleaning function
def clean_data(df, missing_threshold=0.5, zscore_threshold=3, numeric_range=None):
    explanation_steps = []
    
    # Step 1: Drop columns or rows with excessive missing values
    missing_col_percent = df.isnull().mean()  # Percentage of missing values per column
    initial_shape = df.shape
    df = df.loc[:, missing_col_percent < missing_threshold]
    if df.shape != initial_shape:
        explanation_steps.append("Step 1: Removed columns with excessive missing values.")
    else:
        explanation_steps.append("Step 1: No columns with excessive missing values found.")
    
    missing_row_percent = df.isnull().mean(axis=1)  # Percentage of missing values per row
    df = df.loc[missing_row_percent < missing_threshold]
    if df.shape != initial_shape:
        explanation_steps.append("Step 1: Removed rows with excessive missing values.")
    else:
        explanation_steps.append("Step 1: No rows with excessive missing values found.")

    if "Name" in df.columns:
        df = df[df["Name"].notna()]  # Remove rows where 'Name' is NaN
        df = df[df["Name"] != ""]  # Remove rows where 'Name' is an empty string
        explanation_steps.append("Step 1: Removed rows where 'Name' was NaN or empty.")
    else:
        explanation_steps.append("Step 1: 'Name' column not found or no action needed.")

    # Step 2: Handling Text in Number Columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in df.columns.difference(numeric_cols):
        # Check if majority of values are numeric
        if df[col].astype(str).str.isnumeric().mean() > 0.5:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            explanation_steps.append(f"Step 2: Converted text in column '{col}' to numeric.")
        else:
            explanation_steps.append(f"Step 2: No text to convert in column '{col}'.")

    # Step 3: Handling missing values
    for col in df.select_dtypes(include=["number"]).columns:
        missing_values = df[col].isna().sum()  # Count missing values in the column
        if missing_values > 0:
            mean_value = df[col].mean()  # Calculate the mean
            rounded_mean = np.ceil(mean_value)  # Round up the mean
            df[col].fillna(rounded_mean, inplace=True)  # Fill missing values with the rounded-up mean
            explanation_steps.append(f"Step 3: Filled {missing_values} missing values in numeric column '{col}' with rounded mean ({rounded_mean}).")
        else:
            explanation_steps.append(f"Step 3: No missing values in numeric column '{col}'.")

    # Process "Name" and "name" columns
    for col in ["Name", "name"]:
        if col in df.columns:
            # Replace missing or empty names with "Unknown"
            df[col] = df[col].fillna("Unknown")
            df[col] = df[col].replace("", "Unknown")
            explanation_steps.append(f"Step 4: Replaced missing or empty names in column '{col}' with 'Unknown'.")
            # Replace names containing numbers with "Unknown"
            df[col] = df[col].apply(lambda x: "Unknown" if any(char.isdigit() for char in x) else x)
        else:
            explanation_steps.append(f"Step 4: '{col}' column not found or no action needed.")

    # Example: Fill missing values for other categorical columns with mode
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col not in ["Name", "name"]:  # Exclude the 'Name' column
            mode_value = df[col].mode()[0]  # Get the mode of the column
            df[col].fillna(mode_value, inplace=True)  # Fill missing values with the mode
            explanation_steps.append(f"Step 5: Filled missing values in categorical column '{col}' with mode.")
        else:
            explanation_steps.append(f"Step 5: No action needed for categorical column '{col}'.")

    # Step 4: Outlier Detection and Handling
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col]))
        if any(z_scores > zscore_threshold):
            df[col] = np.where(
                z_scores > zscore_threshold,
                np.where(df[col] > df[col].mean(), df[col].quantile(0.95), df[col].quantile(0.05)),
                df[col]
            )
            explanation_steps.append(f"Step 6: Handled outliers in column '{col}' using Z-score.")
        else:
            explanation_steps.append(f"Step 6: No outliers detected in column '{col}'.")

    # Step 5: Handle out-of-range values
    if numeric_range:
        for col, (min_val, max_val) in numeric_range.items():
            if col in df.columns:
                df[col] = np.clip(df[col], min_val, max_val)
                explanation_steps.append(f"Step 7: Clipped values in column '{col}' to range ({min_val}, {max_val}).")
            else:
                explanation_steps.append(f"Step 7: Column '{col}' not found, no action taken.")
    else:
        explanation_steps.append("Step 7: No numeric range specified, no action taken.")

    # Step 6: Handle duplicate rows
    initial_row_count = df.shape[0]
    df = df.drop_duplicates()
    if df.shape[0] != initial_row_count:
        explanation_steps.append("Step 8: Removed duplicate rows.")
    else:
        explanation_steps.append("Step 8: No duplicate rows found.")

    return df, explanation_steps

# Function to generate the report with charts and explanations
def generate_report(df, chart, explanation_steps):
    # Create an Excel report in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # 1. Write the cleaned data to the "Cleaned Data" sheet
        df.to_excel(writer, index=False, sheet_name="Cleaned Data")

        # 2. Add the explanation to the "Explanation" sheet
        workbook = writer.book
        worksheet = workbook.create_sheet("Explanation")
        
        explanation_text = ["Data Cleaning Report", "----------------------"] + explanation_steps
        
        for idx, line in enumerate(explanation_text):
            worksheet.append([line])

        # 3. Add the chart to the "Chart" sheet
        chart_image = chart.to_image(format="png")
        chart_image_stream = io.BytesIO(chart_image)

        # Create a new sheet for the chart
        worksheet = workbook.create_sheet("Chart")
        # Add the image to the new sheet
        worksheet.add_image(openpyxl.drawing.image.Image(chart_image_stream), "A1")
    
    # Save the file to memory and return the bytes
    output.seek(0)
    return output


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

        # Automatically clean data when uploaded
        df, explanation_steps = clean_data(df)
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
            all_columns = df.columns.tolist()  # Include all columns, not just numeric
            if len(all_columns) < 1:
                st.warning("No columns available for visualization.")
                return

            x_axis = st.selectbox("X-Axis", options=all_columns)
            y_axis = st.selectbox("Y-Axis", options=all_columns)

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
            report = generate_report(df, fig, explanation_steps)
            st.download_button(
                label="Download Excel Report with Explanation and Chart",
                data=report,
                file_name="Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
