# eda.py
import streamlit as st
import pandas as pd
import os
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')


# Title
st.title("Pandas Profiling Report")
# get the uploaded or selected dataset path
path = st.session_state.get("dataset_path", None)

if not path or not os.path.exists(path):
    st.warning("Please upload or select a dataset on the Home / Import Data page first.")
    st.stop()

# show a small preview
df = pd.read_csv(path)
# st.write("Preview of your data:", df.head(3))

# when clicked, generate & embed the full report
if st.button("Generate Full‑Page Profiling Report"):
    with st.spinner("Generating profiling report…"):
        pr = ProfileReport(df, explorative=True)

        # 2) convert to HTML and embed as a full‑page scrollable component
        html = pr.to_html()
        components.html(
            html,
            height=st.session_state.get("report_height", 1000), 
            scrolling=True,
        )

if st.button("in-page profiling"):
    # Prepare column lists by type
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    bool_cols = df.select_dtypes(include='bool').columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    # Create tabs for sections
    tabs = st.tabs([
        "Overview", 
        "Variables", 
        "Interactions", 
        "Correlations", 
        "Missing Values", 
        "Sample", 
        "Duplicates", 
        "Alerts"
    ])

    # 1. Overview
    with tabs[0]:
        st.header("Overview")
        # Dataset dimensions
        rows, cols = df.shape
        st.write(f"**Dimensions:** {rows} rows, {cols} columns")
        # Missing values total
        total_missing = int(df.isnull().sum().sum())
        st.write(f"**Missing cells:** {total_missing}")
        # Duplicate rows
        duplicate_count = int(df.duplicated(keep=False).sum())
        st.write(f"**Duplicate rows:** {duplicate_count}")
        # Memory usage
        mem_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)
        st.write(f"**Memory usage:** {mem_usage:.2f} MB")
        # Data types
        st.write("**Data Types:**")
        dtype_df = pd.DataFrame(df.dtypes, columns=['Type']).reset_index().rename(columns={'index': 'Column'})
        st.dataframe(dtype_df)
        # Brief alerts summary (flag if any)
        constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
        high_card_cols = [col for col in categorical_cols if df[col].nunique() > 50]
        corr = df[numeric_cols].corr() if numeric_cols else pd.DataFrame()
        perfect_pairs = []
        if not corr.empty:
            for i in range(len(corr.columns)):
                for j in range(i):
                    if abs(corr.iloc[i, j]) > 0.9999:
                        perfect_pairs.append((corr.columns[i], corr.columns[j]))
        if constant_cols or high_card_cols or perfect_pairs:
            st.write("**Alerts:** Potential issues detected (see Alerts tab).")

    # 2. Variable Summary
    with tabs[1]:
        st.header("Variable Summary")
        # Numeric variables
        if numeric_cols:
            st.subheader("Numeric Variables")
            for col in numeric_cols:
                st.write(f"**{col}**")
                stats = {
                    'Count': int(df[col].count()),
                    'Unique': int(df[col].nunique()),
                    'Missing': int(df[col].isnull().sum()),
                    'Mean': float(df[col].mean()) if df[col].count() > 0 else np.nan,
                    'Std': float(df[col].std()) if df[col].count() > 0 else np.nan,
                    'Min': float(df[col].min()) if df[col].count() > 0 else np.nan,
                    '25%': float(df[col].quantile(0.25)) if df[col].count() > 0 else np.nan,
                    '50%': float(df[col].median()) if df[col].count() > 0 else np.nan,
                    '75%': float(df[col].quantile(0.75)) if df[col].count() > 0 else np.nan,
                    'Max': float(df[col].max()) if df[col].count() > 0 else np.nan
                }
                stats_df = pd.DataFrame(stats.items(), columns=['Statistic', 'Value'])
                st.table(stats_df.set_index('Statistic'))
                fig, ax = plt.subplots()
                sns.histplot(df[col].dropna(), ax=ax, kde=True, color='skyblue')
                ax.set_title(f"Distribution of {col}")
                ax.set_xlabel(col)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        # Categorical variables
        if categorical_cols:
            st.subheader("Categorical Variables")
            for col in categorical_cols:
                st.write(f"**{col}**")
                value_counts = df[col].value_counts(dropna=False)
                vc_df = pd.DataFrame({'Count': value_counts})
                st.table(vc_df)
                fig, ax = plt.subplots()
                top_counts = value_counts.head(10)
                sns.barplot(x=top_counts.values, y=top_counts.index.astype(str), ax=ax, palette='viridis')
                ax.set_title(f"Top values in {col}")
                ax.set_xlabel("Count")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        # Boolean variables
        if bool_cols:
            st.subheader("Boolean Variables")
            for col in bool_cols:
                st.write(f"**{col}**")
                bool_counts = df[col].value_counts(dropna=False)
                bvc_df = pd.DataFrame({'Count': bool_counts})
                st.table(bvc_df)
                fig, ax = plt.subplots()
                sns.barplot(x=bool_counts.index.astype(str), y=bool_counts.values, ax=ax, palette='pastel')
                ax.set_title(f"Counts of {col}")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        # Datetime variables
        if datetime_cols:
            st.subheader("Datetime Variables")
            for col in datetime_cols:
                st.write(f"**{col}**")
                dt_col = df[col]
                stats = {
                    'Count': int(dt_col.count()),
                    'Unique': int(dt_col.nunique()),
                    'Missing': int(dt_col.isnull().sum()),
                    'First': dt_col.min(),
                    'Last': dt_col.max()
                }
                stats_df = pd.DataFrame(stats.items(), columns=['Statistic', 'Value'])
                st.table(stats_df.set_index('Statistic'))
                fig, ax = plt.subplots()
                clean_dates = dt_col.dropna()
                if not clean_dates.empty:
                    years = clean_dates.dt.year
                    year_counts = years.value_counts().sort_index()
                    sns.barplot(x=year_counts.index.astype(str), y=year_counts.values, ax=ax, color='salmon')
                    ax.set_title(f"Count by Year for {col}")
                    ax.set_xlabel("Year")
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close(fig)

    # 3. Interactions
    with tabs[2]:
        st.header("Interactions")
        if numeric_cols and len(numeric_cols) > 1:
            st.write("Pairwise scatterplot matrix for numeric variables:")
            sample_df = df[numeric_cols].dropna()
            if len(sample_df) > 500:
                sample_df = sample_df.sample(500, random_state=1)
            axes = pd.plotting.scatter_matrix(sample_df, figsize=(10, 10), diagonal='hist', color='orange')
            plt.suptitle("Scatterplot Matrix", y=1.02)
            fig = plt.gcf()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.write("Not enough numeric columns for interaction plots.")

    # 4. Correlations
    with tabs[3]:
        st.header("Correlations")
        if numeric_cols:
            corr_methods = ['pearson', 'spearman', 'kendall']
            for method in corr_methods:
                st.subheader(f"{method.capitalize()} Correlation")
                corr = df[numeric_cols].corr(method=method)
                size = len(numeric_cols)
                if size < 3:
                    size = 3
                if size > 15:
                    size = 15
                fig, ax = plt.subplots(figsize=(size, size))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
                ax.set_title(f"{method.capitalize()} correlation")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.write("No numeric columns to calculate correlations.")

    # 5. Missing Values
    with tabs[4]:
        st.header("Missing Values")
        if df.isnull().values.any():
            st.subheader("Missing Values Matrix")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False, ax=ax)
            ax.set_title("Missing Values (blank = missing)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.subheader("Missing Values Correlation")
            size = len(df.columns)
            if size < 3:
                size = 3
            if size > 15:
                size = 15
            fig, ax = plt.subplots(figsize=(size, size))
            miss_corr = df.isnull().astype(int).corr()
            sns.heatmap(miss_corr, annot=True, cmap='viridis', ax=ax)
            ax.set_title("Correlation of Missing Values")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.write("No missing values in the dataset.")

    # 6. Sample
    with tabs[5]:
        st.header("Sample")
        st.write("Random sample of 5 rows:")
        st.dataframe(df.sample(min(5, len(df)), random_state=1))

    # 7. Duplicates
    with tabs[6]:
        st.header("Duplicates")
        dups = df[df.duplicated(keep=False)]
        if not dups.empty:
            st.write(f"Number of duplicate rows: {len(dups)}")
            st.dataframe(dups)
        else:
            st.write("No duplicate rows found.")

    # 8. Alerts
    with tabs[7]:
        st.header("Alerts")
        issues = []
        # Constant columns
        const_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
        if const_cols:
            issues.append(f"Constant columns: {', '.join(const_cols)}")
        # High cardinality categorical columns
        high_card_cols = [col for col in categorical_cols if df[col].nunique() > 50]
        if high_card_cols:
            issues.append(f"High cardinality columns (>50 unique): {', '.join(high_card_cols)}")
        # Perfect correlations among numeric columns
        if numeric_cols:
            corr_abs = df[numeric_cols].corr().abs()
            perfect_corr_pairs = []
            cols = corr_abs.columns
            for i in range(len(cols)):
                for j in range(i):
                    if corr_abs.iloc[i, j] > 0.9999:
                        perfect_corr_pairs.append(f"{cols[i]} & {cols[j]}")
            if perfect_corr_pairs:
                issues.append(f"Perfectly correlated pairs: {', '.join(perfect_corr_pairs)}")
        if issues:
            for issue in issues:
                st.markdown(f"- {issue}")
        else:
            st.write("No alerts detected.")