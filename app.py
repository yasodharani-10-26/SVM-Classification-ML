import streamlit as st
import kagglehub
import os

st.set_page_config(
    page_title="SVM Classification",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 SVM Classification")
st.write("Loading the Kaggle dataset...")

try:
    dataset_path = kagglehub.dataset_download(
        "vinod00725/svm-classification"
    )

    st.success("✅ Dataset downloaded successfully!")

    st.write("### 📂 Dataset Files")

    files = os.listdir(dataset_path)

    for file in files:
        st.write("📄", file)

except Exception as e:
    st.error(f"❌ Unable to load dataset: {e}")
