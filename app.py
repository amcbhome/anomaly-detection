import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# ──────────────────────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ANOVA Anomaly Detection",
    layout="centered"
)
st.title("📊 ANOVA-based Anomaly Detection in Monthly Sales")

st.markdown("""
This descriptive analytics tool applies **Analysis of Variance (ANOVA)**  
to detect statistically significant differences between monthly averages.  

Upload a dataset (for example, the Kaggle *Superstore Sales* or *Sample Sales Data*),  
and the app will test for anomalies in sales or profit trends.
""")

# ──────────────────────────────────────────────────────────────
# Step 1 – File Upload
# ──────────────────────────────────────────────────────────────
st.subheader("Step 1 – Upload or Load Dataset")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully.")
else:
    st.info("No file uploaded. Loading sample dataset from Kaggle's Sample Sales Data...")
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/SalesKaggle3.csv"
    df = pd.read_csv(url)

st.write("### Preview of Dataset")
st.dataframe(df.head(), use_container_width=True)

# ──────────────────────────────────────────────────────────────
# Step 2 – Select Variables
# ──────────────────────────────────────────────────────────────
st.subheader("Step 2 – Select Columns for Analysis")

# Try to auto-detect likely columns
date_cols = [c for c in df.columns if "date" in c.lower() or "order" in c.lower()]
numeric_cols = df.select_dtypes(include="number").columns.tolist()

date_col = st.selectbox("Select date column:", date_cols if date_cols else df.columns)
value_col = st.selectbox("Select numeric column (e.g. Sales, Profit):", numeric_cols)

# ──────────────────────────────────────────────────────────────
# Step 3 – Prepare Data
# ──────────────────────────────────────────────────────────────
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df["Month"] = df[date_col].dt.month
df["MonthName"] = df[date_col].dt.strftime("%b")

# Drop missing or invalid dates
df = df.dropna(subset=["Month", value_col])

# ──────────────────────────────────────────────────────────────
# Step 4 – Perform ANOVA
# ──────────────────────────────────────────────────────────────
st.subheader("Step 3 – Run ANOVA Test")

groups = [df[df["Month"] == m][value_col] for m in sorted(df["Month"].unique())]
f_stat, p_value = f_oneway(*groups)

st.write(f"**F-Statistic:** {f_stat:.3f}")
st.write(f"**p-Value:** {p_value:.5f}")

if p_value < 0.05:
    st.error("⚠️ Significant differences detected between months → possible anomalies.")
else:
    st.success("✅ No significant differences detected between monthly averages.")

# ──────────────────────────────────────────────────────────────
# Step 5 – Visualise
# ──────────────────────────────────────────────────────────────
st.subheader("Step 4 – Visualisation")

fig, ax = plt.subplots(figsize=(7, 5))
df.boxplot(column=value_col, by="MonthName", ax=ax, grid=False, color="black")
plt.suptitle("")
ax.set_title(f"Monthly Variance in {value_col} (ANOVA View)")
ax.set_xlabel("Month")
ax.set_ylabel(value_col)
st.pyplot(fig)

# ──────────────────────────────────────────────────────────────
# Step 6 – Summary Interpretation
# ──────────────────────────────────────────────────────────────
st.subheader("Step 5 – Interpretation")

if p_value < 0.05:
    st.markdown(
        f"""
        The **p-value = {p_value:.4f}** indicates at least one month’s mean differs significantly  
        from others. This suggests potential anomalies or seasonality effects in **{value_col}**.  
        You can review the boxplot above to identify outlier months.
        """
    )
else:
    st.markdown(
        f"""
        The **p-value = {p_value:.4f}** shows no significant variation between months.  
        Monthly averages of **{value_col}** appear stable, with no strong anomalies detected.
        """
    )

st.caption("Generated for portfolio demonstration – Descriptive Analytics (ANOVA).")
