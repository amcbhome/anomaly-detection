import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Automated ANOVA Anomaly Detection",
    layout="centered"
)

st.title("📊 Automated ANOVA-Based Anomaly Detection")

st.markdown("""
This **descriptive analytics tool** automatically detects anomalies in your dataset  
by applying *Analysis of Variance (ANOVA)* to test if group means differ significantly.

It can automatically select a **label (categorical)** and **value (numeric)** column  
to make the process adaptive and data-driven.
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
# Step 2 – Auto-detect label/value columns
# ─────────────────────────────────────────────
st.subheader("Step 2 – Column Detection")

numeric_cols = df.select_dtypes(include="number").columns.tolist()
categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

auto_label_col = None
auto_value_col = None

for col in categorical_cols:
    if df[col].nunique() > 1 and df[col].nunique() < 20:
        auto_label_col = col
        break

for col in numeric_cols:
    if df[col].notna().sum() > 10:
        auto_value_col = col
        break

if not auto_label_col or not auto_value_col:
    st.warning("⚠️ Could not auto-detect suitable columns. Please select manually.")

label_col = st.selectbox("Select or confirm the label (categorical) column:", 
                         categorical_cols, 
                         index=categorical_cols.index(auto_label_col) if auto_label_col in categorical_cols else 0)
value_col = st.selectbox("Select or confirm the numeric column:", 
                         numeric_cols, 
                         index=numeric_cols.index(auto_value_col) if auto_value_col in numeric_cols else 0)

st.markdown(f"**Analysing:** `{value_col}` grouped by `{label_col}`")

# ─────────────────────────────────────────────
# Step 3 – Run ANOVA
# ─────────────────────────────────────────────
st.subheader("Step 3 – Run ANOVA Test")

groups = [g[value_col].dropna() for _, g in df.groupby(label_col) if len(g) > 1]

if len(groups) < 2:
    st.warning("⚠️ Not enough valid groups to run ANOVA. Select different columns.")
    st.stop()

f_stat, p_value = f_oneway(*groups)

st.write(f"**F-Statistic:** {f_stat:.3f}")
st.write(f"**p-Value:** {p_value:.5f}")

if p_value < 0.05:
    st.error("⚠️ Significant differences detected → possible anomalies between groups.")
else:
    st.success("✅ No significant variance detected between selected groups.")

# ─────────────────────────────────────────────
# Step 4 – Show Group Means
# ─────────────────────────────────────────────
st.subheader("Step 4 – Group Means (for anomaly inspection)")
means = df.groupby(label_col)[value_col].agg(['mean', 'std', 'count']).round(2)
st.dataframe(means)

# ─────────────────────────────────────────────
# Step 5 – Visualise
# ─────────────────────────────────────────────
st.subheader("Step 5 – Visualisation")

fig, ax = plt.subplots(figsize=(8, 5))
df.boxplot(column=value_col, by=label_col, ax=ax, grid=False)
plt.suptitle("")
ax.set_title(f"Variance in {value_col} by {label_col} (ANOVA View)")
ax.set_xlabel(label_col)
ax.set_ylabel(value_col)
st.pyplot(fig)

# ─────────────────────────────────────────────
# Step 6 – Post-hoc Analysis (Tukey HSD)
# ─────────────────────────────────────────────
if p_value < 0.05:
    st.subheader("Step 6 – Post-hoc Comparison (Tukey HSD)")
    try:
        tukey = pairwise_tukeyhsd(endog=df[value_col], groups=df[label_col], alpha=0.05)
        st.text(tukey.summary())
        st.caption("✅ 'Reject = True' indicates a statistically significant difference between those groups.")
    except Exception as e:
        st.warning("Tukey test could not be computed. Ensure groups have multiple observations each.")
else:
    st.info("Tukey post-hoc test not required since ANOVA found no significant differences.")

# ─────────────────────────────────────────────
# Step 7 – Interpretation
# ─────────────────────────────────────────────
st.subheader("Step 7 – Interpretation")

if p_value < 0.05:
    st.markdown(f"""
    The **p-value = {p_value:.4f}** indicates that at least one group's mean differs significantly.  
    This suggests an anomaly or unusual variation in **{value_col}** across **{label_col}**.  
    Check the boxplot and the Tukey HSD output above to identify which groups are significantly different.
    """)
else:
    st.markdown(f"""
    The **p-value = {p_value:.4f}** suggests no statistically significant differences  
    between the group averages. **{value_col}** appears stable across **{label_col}**.
    """)

st.caption("Generated for portfolio demonstration – Descriptive Analytics (Automated ANOVA).")

