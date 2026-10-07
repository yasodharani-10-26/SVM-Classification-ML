import streamlit as st
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

st.set_page_config(
    page_title="SVM Classification",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 SVM Classification")
st.subheader("Universal Bank Classification")

try:
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "vinod00725/svm-classification",
        "UniversalBank.csv"
    )

    st.success("✅ Dataset loaded successfully!")

except Exception as e:
    st.error(f"❌ Unable to load dataset: {e}")
    st.stop()

# Dataset information
st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📋 Records", f"{len(df):,}")

with col2:
    st.metric("🔢 Features", f"{len(df.columns):,}")

with col3:
    st.metric(
        "❌ Missing Values",
        f"{df.isnull().sum().sum():,}"
    )

st.divider()

# Dataset preview
st.header("📄 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)

# Columns
st.header("🔎 Dataset Columns")

for i, column in enumerate(df.columns, 1):
    st.write(f"{i}. `{column}`")

# Data types
st.header("🧾 Data Types")

column_info = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Missing Values": df.isnull().sum().values,
    "Unique Values": [
        df[column].nunique()
        for column in df.columns
    ]
})

st.dataframe(
    column_info,
    use_container_width=True
)
