import streamlit as st
import pandas as pd
from scipy.stats import f_oneway

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="Automated ANOVA Sweep", layout="centered")
st.title("📊 Automated ANOVA Sweep – Descriptive Analytics")

st.markdown("""
This app automatically tests **all combinations of categorical and numeric columns**  
in the dataset using **ANOVA (Analysis of Variance)** to detect possible anomalies.

It flags any relationships where the means differ significantly (**p < 0.05**).
""")

# ─────────────────────────────────────────────
# Step 1 – Load dataset
# ─────────────────────────────────────────────
st.subheader("Step 1 – Load Dataset")

url = "https://raw.githubusercontent.com/amcbhome/anomaly-detection/main/Multiclass%20Clothing%20Sales%20Dataset.csv"
st.markdown(f"**Dataset Source:** [GitHub Dataset]({url})")

try:
    df = pd.read_csv(url)
    st.success("✅ Dataset loaded successfully from GitHub.")
except Exception:
    st.error("⚠️ Unable to load dataset automatically.")
    uploaded_file = st.file_uploader("Upload CSV manually:", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully.")
    else:
        st.stop()

st.write("### Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# ─────────────────────────────────────────────
# Step 2 – Identify column types
# ─────────────────────────────────────────────
st.subheader("Step 2 – Identify Categorical and Numeric Columns")

categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
numeric_cols = df.select_dtypes(include="number").columns.tolist()

st.markdown(f"**Categorical columns:** {len(categorical_cols)} detected → {', '.join(categorical_cols[:10])}")
st.markdown(f"**Numeric columns:** {len(numeric_cols)} detected → {', '.join(numeric_cols[:10])}")

if not categorical_cols or not numeric_cols:
    st.error("No suitable categorical or numeric columns found.")
    st.stop()

# ─────────────────────────────────────────────
# Step 3 – Automated ANOVA Sweep
# ─────────────────────────────────────────────
st.subheader("Step 3 – Automated ANOVA Sweep")

results = []

for cat_col in categorical_cols:
    if df[cat_col].nunique() < 2 or df[cat_col].nunique() > 20:
        continue  # skip unsuitable grouping columns
    for num_col in numeric_cols:
        sub_df = df[[cat_col, num_col]].dropna()
        groups = [g[num_col].values for _, g in sub_df.groupby(cat_col) if len(g) > 1]

        if len(groups) < 2:
            continue

        try:
            f_stat, p_value = f_oneway(*groups)
            results.append({
                "Category": cat_col,
                "Numeric_Variable": num_col,
                "F-Statistic": round(f_stat, 3),
                "p-Value": round(p_value, 5),
                "Significant": "✅" if p_value < 0.05 else "❌"
            })
        except Exception:
            continue

# ─────────────────────────────────────────────
# Step 4 – Display Results
# ─────────────────────────────────────────────
if results:
    results_df = pd.DataFrame(results)
    st.write("### ANOVA Summary Results")
    st.dataframe(results_df, use_container_width=True)

    sig = results_df[results_df["Significant"] == "✅"]
    if not sig.empty:
        st.success(f"Detected {len(sig)} significant relationships (p < 0.05).")
        st.dataframe(sig)
    else:
        st.info("No statistically significant anomalies detected across column pairs.")
else:
    st.warning("No valid ANOVA results found.")

# ─────────────────────────────────────────────
# Step 5 – Interpretation
# ─────────────────────────────────────────────
st.subheader("Step 4 – Interpretation")

st.markdown("""
The automated sweep tested each categorical column against every numeric column.  
Where **p < 0.05**, at least one group mean differs significantly — a potential anomaly.

This approach mimics a **data assurance scan**, automatically flagging unusual  
relationships between categories and numeric metrics, ideal for large datasets.
""")

st.caption("Generated for portfolio use – Automated Descriptive Analytics (ANOVA Sweep).")


