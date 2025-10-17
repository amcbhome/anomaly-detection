import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ANOVA-Based Anomaly Detection",
    layout="centered"
)

st.title("📊 ANOVA-Based Anomaly Detection in Clothing Sales")

st.markdown("""
This **descriptive analytics tool** uses *Analysis of Variance (ANOVA)*  
to detect statistically significant differences between groups —  
for example, between months, product categories, or regions.

It uses the **Dynamic Apparel Sales Dataset with Anomalies** (Kaggle)  
hosted in this GitHub repository for reproducibility.
""")

# ─────────────────────────────────────────────
# Step 1 – Load dataset
# ─────────────────────────────────────────────
st.subheader("Step 1 – Load Dataset")

url = "https://raw.githubusercontent.com/amcbhome/anomaly-detection/main/Multiclass%20Clothing%20Sales%20Dataset.csv"
st.markdown(f"**Source:** [Multiclass Clothing Sales Dataset (GitHub)]({url})")

try:
    df = pd.read_csv(url)
    st.success("✅ Successfully loaded dataset from GitHub.")
except Exception:
    st.error("⚠️ Could not load dataset from GitHub. Please upload manually below.")
    uploaded_file = st.file_uploader("Upload CSV file manually:", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully.")
    else:
        st.stop()

st.write("### Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# ─────────────────────────────────────────────
# Step 2 – Select Columns
# ─────────────────────────────────────────────
st.subheader("Step 2 – Select Columns")

# Auto-detect likely column types
date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
numeric_cols = df.select_dtypes(include="number").columns.tolist()
category_cols = df.select_dtypes(exclude="number").columns.tolist()

if date_cols:
    group_col = st.selectbox("Select date column for grouping (Month will be extracted):", date_cols)
    group_type = "date"
else:
    group_col = st.selectbox("Select categorical column for grouping:", category_cols)
    group_type = "category"

value_col = st.selectbox("Select numeric column (e.g. Sales, Revenue, Price):", numeric_cols)

# ─────────────────────────────────────────────
# Step 3 – Prepare Data
# ─────────────────────────────────────────────
if group_type == "date":
    df[group_col] = pd.to_datetime(df[group_col], errors="coerce")
    df["GroupLabel"] = df[group_col].dt.strftime("%b")  # Month abbreviations
else:
    df["GroupLabel"] = df[group_col].astype(str)

df = df.dropna(subset=["GroupLabel", value_col])

# ─────────────────────────────────────────────
# Step 4 – Run ANOVA
# ─────────────────────────────────────────────
st.subheader("Step 3 – Run ANOVA Test")

# Group data and check validity
groups = [g[value_col].dropna() for _, g in df.groupby("GroupLabel") if len(g) > 1]

if len(groups) < 2:
    st.warning("⚠️ Not enough valid groups to run ANOVA. Select a different column or ensure numeric data is present.")
else:
    f_stat, p_value = f_oneway(*groups)
    st.write(f"**F-Statistic:** {f_stat:.3f}")
    st.write(f"**p-Value:** {p_value:.5f}")

    if p_value < 0.05:
        st.error("⚠️ Significant differences detected between groups → possible anomalies or variance.")
    else:
        st.success("✅ No significant variance detected between selected groups.")

# ─────────────────────────────────────────────
# Step 5 – Visualisation
# ─────────────────────────────────────────────
st.subheader("Step 4 – Visualisation")

if len(groups) >= 2:
    fig, ax = plt.subplots(figsize=(8, 5))
    df.boxplot(column=value_col, by="GroupLabel", ax=ax, grid=False)
    plt.suptitle("")
    ax.set_title(f"Variance in {value_col} by {group_col} (ANOVA View)")
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)
    st.pyplot(fig)

# ─────────────────────────────────────────────
# Step 6 – Interpretation
# ─────────────────────────────────────────────
st.subheader("Step 5 – Interpretation")

if len(groups) < 2:
    st.info("Upload a dataset with a valid grouping column and numeric field to run ANOVA.")
else:
    if p_value < 0.05:
        st.markdown(f"""
        The **p-value = {p_value:.4f}** indicates that at least one group’s mean differs significantly.  
        This suggests an anomaly or unusual variation in **{value_col}** across **{group_col}**.  
        Review the boxplot above to identify categories with outlier values.
        """)
    else:
        st.markdown(f"""
        The **p-value = {p_value:.4f}** suggests no statistically significant differences  
        between the group averages. **{value_col}** appears stable across **{group_col}**.
        """)

st.caption("Generated for portfolio demonstration – Descriptive Analytics (ANOVA).")

