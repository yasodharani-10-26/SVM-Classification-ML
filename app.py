import streamlit as st
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SVM Classification",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🤖 SVM Classification")
st.subheader("Machine Learning Classification Dashboard")

st.write(
    "This project uses a Support Vector Machine (SVM) "
    "classification algorithm to analyze the Universal Bank dataset."
)

st.divider()

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

FILE_PATH = "UniversalBank.csv"

try:

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "vinod00725/svm-classification",
        FILE_PATH
    )

    st.success("✅ Dataset loaded successfully!")

except Exception as e:

    st.error(f"❌ Unable to load dataset: {e}")
    st.stop()

# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "📋 Records",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "🔢 Features",
        f"{len(df.columns):,}"
    )

with col3:
    st.metric(
        "❌ Missing Values",
        f"{df.isnull().sum().sum():,}"
    )

# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------

st.subheader("📄 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)

# --------------------------------------------------
# COLUMN INFORMATION
# --------------------------------------------------

st.subheader("🔎 Dataset Columns")

st.write(list(df.columns))

# --------------------------------------------------
# DATA TYPES
# --------------------------------------------------

st.subheader("🧾 Data Types")

st.dataframe(
    pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    }),
    use_container_width=True
)
