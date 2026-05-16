import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport

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

        numeric_columns = df.select_dtypes(include=['number']).columns

        profile = ProfileReport(df, explorative=True)

        tab1, tab2, tab3, tab4 = st.tabs([
            "📄 Preview",
            "📊 Summary",
            "📈 Visualizations",
            "🧠 EDA Report"
        ])

        # ---------------- TAB 1 ----------------

        with tab1:

            st.subheader("Dataset Preview")
            st.dataframe(df.head())

        # ---------------- TAB 2 ----------------

        with tab2:

            st.subheader("Dataset Summary")

            col1, col2, col3 = st.columns(3)

            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing Values", df.isnull().sum().sum())

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

        # ---------------- TAB 3 ----------------

        with tab3:

            st.subheader("Custom Correlation Analysis")

            show_values = st.checkbox("Show Correlation Values")

            corr_matrix = df.corr(numeric_only=True)

            fig, ax = plt.subplots(figsize=(14, 8))

            sns.heatmap(
                corr_matrix,
                annot=show_values,
                fmt=".2f",
                cmap="coolwarm",
                linewidths=0.5,
                annot_kws={"size": 8},
                ax=ax
            )

            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)

            st.pyplot(fig)

            st.subheader("Column Distribution")

            if len(numeric_columns) > 0:

                selected_column = st.selectbox(
                    "Select Column for Histogram",
                    numeric_columns
                )

                fig2, ax2 = plt.subplots(figsize=(4, 2))

                sns.histplot(df[selected_column], kde=True, ax=ax2)

                st.pyplot(fig2, use_container_width=False)

                st.subheader("Box Plot")

                if len(numeric_columns) > 0:
                    box_column = st.selectbox(
                        "Select Column for Box Plot",
                        numeric_columns,
                        key="boxplot"
                    )

                    fig3, ax3 = plt.subplots(figsize=(4, 2))

                    sns.boxplot(x=df[box_column], ax=ax3)

                    st.pyplot(fig3, use_container_width=False)

            st.subheader("Scatter Plot")

            if len(numeric_columns) >= 2:
                x_col = st.selectbox(
                    "Select X-axis",
                    numeric_columns,
                    key="scatter_x"
                )

                y_col = st.selectbox(
                    "Select Y-axis",
                    numeric_columns,
                    key="scatter_y"
                )

                fig4, ax4 = plt.subplots(figsize=(5, 3))

                sns.scatterplot(
                    data=df,
                    x=x_col,
                    y=y_col,
                    ax=ax4
                )

                st.pyplot(fig4, use_container_width=False)

            st.subheader("Count Plot")

            categorical_columns = df.select_dtypes(include=['object']).columns

            if len(categorical_columns) > 0:
                cat_col = st.selectbox(
                    "Select Categorical Column",
                    categorical_columns,
                    key="countplot"
                )

                fig5, ax5 = plt.subplots(figsize=(5, 3))

                sns.countplot(
                    y=df[cat_col],
                    order=df[cat_col].value_counts().index,
                    ax=ax5
                )

                st.pyplot(fig5, use_container_width=False)

        # ---------------- TAB 4 ----------------

        with tab4:

            st.subheader("Full EDA Report")

            st.components.v1.html(
                profile.to_html(),
                height=1000,
                scrolling=True
            )

            profile.to_file("EDA_Report.html")

            with open("EDA_Report.html", "rb") as file:
                st.download_button(
                    label="📥 Download EDA Report",
                    data=file,
                    file_name="EDA_Report.html",
                    mime="text/html"
                )

    except Exception as e:

        st.error(f"Error loading file: {e}")

st.markdown("---")
st.caption("Built by Atish using Streamlit & Python")