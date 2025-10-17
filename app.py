# ─────────────────────────────────────────────
# Step 3 – Automated Multi-Column ANOVA Sweep
# ─────────────────────────────────────────────
st.subheader("Step 3 – Automated ANOVA Sweep")

results = []

for cat_col in categorical_cols:
    if df[cat_col].nunique() < 2 or df[cat_col].nunique() > 20:
        continue  # skip columns not suitable for grouping
    for num_col in numeric_cols:
        # Drop missing and invalid entries
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

# Convert results to DataFrame
if results:
    results_df = pd.DataFrame(results)
    st.write("### ANOVA Summary Table")
    st.dataframe(results_df, use_container_width=True)

    # Filter significant results
    significant_results = results_df[results_df["Significant"] == "✅"]
    if not significant_results.empty:
        st.success(f"Detected {len(significant_results)} significant relationships.")
        st.dataframe(significant_results)
    else:
        st.info("No significant anomalies detected across the available columns.")
else:
    st.warning("No valid ANOVA results found in dataset.")

