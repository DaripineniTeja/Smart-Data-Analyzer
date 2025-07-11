import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def show_dashboard(df):
    st.subheader("📊 Data Dashboard Overview")

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("📄 Rows", df.shape[0])
    col2.metric("📁 Columns", df.shape[1])
    col3.metric("⚠️ Missing Values", int(df.isnull().sum().sum()))

    # Categorical column distribution
    st.markdown("### 🔍 Top Categorical Column Distribution")
    cat_cols = df.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        selected_col = st.selectbox("Select categorical column", cat_cols)
        st.bar_chart(df[selected_col].value_counts().head(10))

    # Correlation heatmap
    st.markdown("### 🧠 Correlation Heatmap")
    num_df = df.select_dtypes(include='number')
    if num_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.info("Need at least 2 numeric columns to show correlation heatmap.")
