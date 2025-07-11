import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv
from openai import OpenAI
from utils import cleaning
from dashboard import show_dashboard
from fpdf import FPDF
from ml_model import ml_model_prediction_tab
import tempfile

from streamlit_lottie import st_lottie

# Load OpenAI API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Page config
st.set_page_config(page_title="Smart Data Analyzer", layout="wide")
st.title("📊 Smart Data Analyzer")
st.markdown("Analyze datasets with visualizations and AI-powered insights.")

# Sidebar file upload
with st.sidebar:
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    use_sample = st.checkbox("Or use sample file (ai_job_trends_dataset.csv)")

# Load data
def load_data():
    if uploaded_file:
        ext = uploaded_file.name.split('.')[-1]
        if ext == "csv":
            return pd.read_csv(uploaded_file)
        elif ext == "xlsx":
            return pd.read_excel(uploaded_file)
    elif use_sample:
        return pd.read_csv("ai_job_trends_dataset.csv")
    return None

df = load_data()
ai_insight_text = ""
chart_fig = None

# Show tabs only if data is loaded
if df is not None:
    tab1, tab2, tab3, tab4, tab5, tab6,tab7,tab8 = st.tabs([
        "📁 Summary", "📊 Charts", "🧹 Cleaning", "🧠 AI Insights", "💬 Ask Your Data", "📄 Export Report as PDF","🤖 ML Prediction","📈 Dashboard"])

    # ---------- TAB 1: Summary ----------
    with tab1:
        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

        st.subheader("📌 Basic Info")
        st.write("Shape:", df.shape)
        st.write("Columns:", df.columns.tolist())

        st.subheader("📊 Statistical Summary")
        st.write(df.describe(include='all'))

        st.subheader("🚨 Null Value Count")
        st.write(df.isnull().sum())

    # ---------- TAB 2: Charts ----------
    with tab2:
        st.subheader("📊 Advanced Visualizations")

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        all_cols = df.columns.tolist()

        st.write("### 🔍 Apply Filters (Optional)")
        filter_col = st.selectbox("Select column to filter (optional)", ["None"] + all_cols)
        if filter_col != "None":
            unique_vals = df[filter_col].dropna().unique().tolist()
            selected_vals = st.multiselect("Select values", unique_vals)
            if selected_vals:
                df = df[df[filter_col].isin(selected_vals)]

        st.write("### 📈 Select Data to Plot")
        x_col = st.selectbox("X-axis", all_cols, index=0)
        y_col = st.selectbox("Y-axis", numeric_cols, index=0)
        chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Box", "Pie"])
        group_by_col = st.selectbox("Group by (optional)", ["None"] + all_cols)

        if st.button("Generate Chart"):
            if group_by_col != "None":
                chart_df = df.groupby(group_by_col)[y_col].mean().reset_index()
                x_col = group_by_col
            else:
                chart_df = df

            if chart_type == "Bar":
                fig = px.bar(chart_df, x=x_col, y=y_col)
            elif chart_type == "Line":
                fig = px.line(chart_df, x=x_col, y=y_col)
            elif chart_type == "Scatter":
                fig = px.scatter(chart_df, x=x_col, y=y_col)
            elif chart_type == "Box":
                fig = px.box(chart_df, x=x_col, y=y_col)
            elif chart_type == "Pie":
                fig = px.pie(chart_df, names=x_col, values=y_col)
            else:
                st.error("Unsupported chart type.")

            chart_fig = fig
            st.plotly_chart(fig, use_container_width=True)

    # ---------- TAB 3: Cleaning ----------
    with tab3:
        st.subheader("🧼 Data Cleaning")

        st.write("### 🔍 Columns with Missing Values")
        null_df = cleaning.get_null_columns(df)
        st.dataframe(null_df.isnull().sum())

        st.write("### 🛠 Fill Missing Values")
        method = st.selectbox("Select fill method", ["mean", "median", "mode"])
        if st.button("Fill Nulls"):
            df = cleaning.fill_missing_values(df, method)
            st.success(f"Missing values filled using {method} method.")

        st.write("### 🧯 Drop Missing Values")
        if st.button("Drop Null Rows"):
            df = cleaning.drop_missing(df)
            st.success("Dropped rows with null values.")

        st.write("### 🔁 Remove Duplicates")
        if st.button("Drop Duplicates"):
            df = cleaning.drop_duplicates(df)
            st.success("Duplicate rows removed.")

        st.write("### 🔄 Change Data Type")
        col_to_convert = st.selectbox("Choose column to convert", df.columns)
        new_dtype = st.selectbox("Choose new type", ["int", "float", "str", "datetime64[ns]"])
        if st.button("Convert Column Type"):
            result = cleaning.change_dtype(df, col_to_convert, new_dtype)
            if isinstance(result, str):
                st.error(result)
            else:
                df = result
                st.success(f"Converted `{col_to_convert}` to `{new_dtype}`.")

        st.write("### ✅ Cleaned Data Preview")
        st.dataframe(df.head())
        st.download_button("📥 Download Cleaned Data", df.to_csv(index=False), file_name="cleaned_data.csv")

    # ---------- TAB 4: AI Insights ----------
    with tab4:
        st.subheader("🧠 AI-Powered Insights")
        prompt = f"Analyze the following dataset and give meaningful job trends or observations:\n\n{df.head(30).to_csv(index=False)}"
        if st.button("Generate AI Insight"):
            with st.spinner("Analyzing with AI..."):
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                ai_insight_text = response.choices[0].message.content
                st.success(ai_insight_text)

    # ---------- TAB 5: Ask Your Data ----------
    with tab5:
        st.subheader("💬 Ask Questions About Your Dataset")

        user_question = st.text_input("Ask anything (e.g., Top 5 job titles by salary):")

        if st.button("🔍 Get Answer"):
            with st.spinner("Thinking like a data analyst..."):
                preview = df.head(15).to_csv(index=False)

                system_prompt = (
                    "You are a data analyst. Based on the user's question and this dataset, "
                    "generate a pandas code snippet using the DataFrame `df` to answer the question. "
                    "Only return code. Assign the result to a variable named `result`. Do not explain."
                )

                user_prompt = f"Dataset Preview:\n{preview}\n\nQuestion: {user_question}"

                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.2
                    )

                    code = response.choices[0].message.content.strip("```python").strip("```")
                    st.code(code, language="python")

                    local_vars = {"df": df.copy()}
                    exec(code, {}, local_vars)

                    if "result" in local_vars:
                        st.success("✅ Here's the result:")
                        st.dataframe(local_vars["result"])
                    else:
                        st.warning("⚠️ No result found. Please ask another question.")

                except Exception as e:
                    st.error(f"❌ Error executing generated code:\n\n{e}")

    # ---------- TAB 6: Export Report ----------
    # ---------- TAB 6: Export Report ----------
    with tab6:
        st.subheader("📄 Export Report as PDF")

    if st.button("Generate PDF Report"):
        try:
            with st.spinner("Generating report..."):
                pdf = FPDF()
                pdf.add_page()

                # Title
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(200, 10, txt="Smart Data Analyzer Report", ln=True, align="C")
                pdf.ln()

                # Dataset Info
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 10, txt="Dataset Information", ln=True)
                pdf.set_font("Arial", size=11)
                pdf.cell(200, 10, txt=f"Total Rows: {df.shape[0]}, Columns: {df.shape[1]}", ln=True)
                pdf.ln()

                # Columns List
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 10, txt="Columns:", ln=True)
                pdf.set_font("Arial", size=11)
                all_columns_text = ", ".join(df.columns.tolist())
                pdf.multi_cell(0, 10, all_columns_text)
                pdf.ln()

                # AI Insights
                if ai_insight_text:
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(200, 10, txt="AI Insights", ln=True)
                    pdf.set_font("Arial", size=11)
                    pdf.multi_cell(0, 10, txt=ai_insight_text)
                    pdf.ln()

                # Chart image
                if chart_fig:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as imgfile:
                        chart_fig.write_image(imgfile.name)
                        pdf.image(imgfile.name, x=10, y=None, w=190)

                # Save & download
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    pdf.output(tmp.name)
                    st.success("✅ PDF Report Generated Successfully!")
                    st.download_button(
                        label="📄 Download PDF",
                        data=open(tmp.name, "rb").read(),
                        file_name="SmartDataReport.pdf",
                        mime="application/pdf"
                    )
        except Exception as e:
            st.error(f"❌ Failed to generate report:\n\n{e}")

    with tab7:
        st.subheader("🤖 Machine Learning - Predictive Modeling")
        ml_model_prediction_tab(df)
    with tab8:
        show_dashboard(df)


else:
    st.warning("📂 Please upload a dataset or select sample file from sidebar.")