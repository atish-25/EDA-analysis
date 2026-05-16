import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AutoEDA", layout="wide")

st.title("📊 AutoEDA Dashboard")

st.markdown("""
Upload any CSV dataset and generate automatic exploratory data analysis reports, visualizations, and statistical insights instantly.
""")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)





        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Dataset Information")

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

        st.subheader("Dataset Summary")

        summary_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.values,
            "Missing Values": df.isnull().sum().values
        })

        numeric_df = df.select_dtypes(include='number')

        summary_df["Mean"] = numeric_df.mean().reindex(df.columns).values
        summary_df["Median"] = numeric_df.median().reindex(df.columns).values
        summary_df["Min"] = numeric_df.min().reindex(df.columns).values
        summary_df["Max"] = numeric_df.max().reindex(df.columns).values

        st.dataframe(summary_df)

        numeric_columns = df.select_dtypes(include=['number']).columns

        if len(numeric_columns) > 0:
            selected_column = st.selectbox(
                "Select Column for Histogram",
                numeric_columns
            )

            fig2, ax2 = plt.subplots(figsize=(5, 3))

            sns.histplot(df[selected_column], kde=True, ax=ax2)

            st.pyplot(fig2, use_container_width=False)



        profile = ProfileReport(df, explorative=True)

        st.subheader("EDA Report")
        st_profile_report(profile)
    except Exception as e:
        st.error(f"Error loading file: {e}")



