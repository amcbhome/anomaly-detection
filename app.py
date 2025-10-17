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
This **descriptive analytics tool** applies *Analysis of Variance (ANOVA)*  
to detect significant differences between monthly sales averages in a dataset.  

It uses the **Dynamic Apparel Sales Dataset with Anomalies** from Kaggle,  
hosted here on your GitHub repository for reproducible analysis.
""")

# ─────────────────────────────────────────────
# Step 1 – Load dataset from GitHub
# ─────────────────────────────────────────────
st.subheader("Step 1 – Load Dataset")

url = "https://raw.githubusercontent.com/amcbhome/anomaly-detection/main/Multiclass%20Clothing%20Sales%20Dataset.csv"
st.markdown(f"**Source:** [Multiclass Clothing Sales Dataset (GitHub)]({url})")

try:
    df = pd.read_csv(url)
    st.success("✅ Successfully loaded dataset from GitHub.")
except Exception as e:
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
# Step 2 – Select columns for analysis
# ─────────────────────────────────────────────
st.subheader("Step 2 – Select Columns")

# Try to detect likely columns
date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
numeric_cols = df.select_dtypes(include="number").columns.tolist()

date_col = st.selectbox("Select date column:", date_cols if date_cols else df.columns)
value_col = st.selectbox("Select numeric column (e.g. Sales, Revenue):", numeric_cols)

# ─────────────────────────────────────────────
# Step 3 – Prepare monthly data
# ─────────────────────────────────────────────
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df["Month"] = df[date_col].dt.month
df["MonthName"] = df[date_col].dt.strftime("%b")
df = df.dropna(subset=["Month", value_col])

# ─────────────────────────────────────────────
# Step 4 – Run ANOVA
# ─────────────────────────────────────────────
st.subheader("Step 3 – Run ANOVA Test")

groups = [df[df["Month"] == m][value_col] for m in sorted(df["Month"].unique())]
f_stat, p_value = f_oneway(*groups)

st.write(f"**F-Statistic:** {f_stat:.3f}")
st.write(f"**p-Value:** {p_value:.5f}")

if p_value < 0.05:
    st.error("⚠️ Significant monthly differences detected → possible anomalies or seasonality.")
else:
    st.success("✅ No significant variance detected between months.")

# ─────────────────────────────────────────────
# Step 5 – Visualise
# ─────────────────────────────────────────────
st.subheader("Step 4 – Visualisation")

fig, ax = plt.subplots(figsize=(8, 5))
df.boxplot(column=value_col, by="MonthName", ax=ax, grid=False)
plt.suptitle("")
ax.set_title(f"Monthly Variance in {value_col} (ANOVA View)")
ax.set_xlabel("Month")
ax.set_ylabel(value_col)
st.pyplot(fig)

# ─────────────────────────────────────────────
# Step 6 – Interpretation
# ─────────────────────────────────────────────
st.subheader("Step 5 – Interpretation")

if p_value < 0.05:
    st.markdown(f"""
    The **p-value = {p_value:.4f}** indicates at least one month's mean differs significantly  
    from the others. This suggests an anomaly or change in behaviour for **{value_col}**.  
    Check the boxplot above for months showing unusually high or low sales.
    """)
else:
    st.markdown(f"""
    The **p-value = {p_value:.4f}** suggests stable monthly averages.  
    No significant anomalies were found in **{value_col}**.
    """)

st.caption("Generated for portfolio demonstration – Descriptive Analytics (ANOVA).")
