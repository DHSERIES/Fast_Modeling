
import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from ImportData import show_import_data
# Title
def show_DataPrep():
    st.title("🔧 Data Preparation and Modeling Training")

    # Load dataset (from session state or upload)
    # get the uploaded or selected dataset path
    path = st.session_state.get("dataset_path", None)
    if not path:
        st.warning("Please upload or select a dataset on the Home / Import Data page first.")
        return show_import_data()

    # show a small preview
    df = pd.read_csv(path)
    st.write(df.head(0))

    target = st.selectbox("Select Target (Label) Column", options=[None] + df.columns.tolist(), key="target_column_select")
    if target:
        st.session_state["target_column"] = target
    else:
        st.warning("Please select a target column to enable training.")
    
    # Identify feature columns by type
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    col1, col2, col3 = st.columns(3)

    # --- Column 1: Missing Value Handling per column ---
    with col1:
        st.subheader("Missing Value Handling")
        auto_missing = st.checkbox("Auto-process missing for all columns", key="auto_missing")
        missing_drop = []
        missing_impute = {}
        if auto_missing:
            # default: impute numeric with mean, categorical with most_frequent
            for col in numeric_cols:
                missing_impute[col] = ("mean", None)
            for col in categorical_cols:
                missing_impute[col] = ("most_frequent", None)
        else:
            for col in numeric_cols + categorical_cols:
                with st.expander(f"{col}"):
                    method = st.selectbox(
                        "Method", ["None", "Drop Rows", "Impute"], key=f"mv_method_{col}"
                    )
                    if method == "Drop Rows":
                        missing_drop.append(col)
                    elif method == "Impute":
                        strategy = st.selectbox(
                            "Strategy", ["mean", "median", "most_frequent", "constant"], key=f"mv_strat_{col}"
                        )
                        fill = None
                        if strategy == "constant":
                            fill = st.text_input("Constant value", value="0", key=f"mv_fill_{col}")
                        missing_impute[col] = (strategy, fill)

    # --- Column 2: Scaling & Encoding per column ---
    with col2:
        st.subheader("Scaling & Encoding")
        auto_preproc = st.checkbox("Auto-scale and encode all columns", key="auto_preproc")
        scale_cols = []
        encode_cols = {}
        if auto_preproc:
            scale_cols = numeric_cols.copy()
            # default encode all categoricals with OneHot
            encode_cols = {col: "OneHot" for col in categorical_cols}
        else:
            for col in numeric_cols:
                with st.expander(f"Scale: {col}"):
                    do_scale = st.checkbox("Scale this column", key=f"scale_{col}")
                    if do_scale:
                        scale_cols.append(col)
            for col in categorical_cols:
                with st.expander(f"Encode: {col}"):
                    do_encode = st.checkbox("Encode this column", key=f"encode_{col}")
                    if do_encode:
                        method = st.selectbox(
                            "Method", ["OneHot", "Ordinal"], key=f"enc_method_{col}"
                        )
                        encode_cols[col] = method

    # --- Column 3: Model Selection & Hyperparameters ---
    with col3:
        st.subheader("Task & Model & Hyperparameters")
        task_type = st.selectbox("Task Type", ["Classification", "Regression"])
        if task_type == "Classification":
            model_choice = st.selectbox("Model", ["RandomForest", "GradientBoosting", "SVM"])
        else:
            model_choice = st.selectbox("Model", ["RandomForestRegressor", "GradientBoostingRegressor", "SVR"])
        params = {}
        if "RandomForest" in model_choice:
            params['n_estimators'] = st.number_input("n_estimators", 10, 500, 100, 10)
            params['max_depth'] = st.number_input("max_depth", 1, 50, 5, 1)
        elif "GradientBoosting" in model_choice:
            params['n_estimators'] = st.number_input("n_estimators", 10, 500, 100, 10)
            params['learning_rate'] = st.number_input("learning_rate", 0.001, 1.0, 0.1, format="%.3f")
        else:
            params['C'] = st.number_input("C", 0.01, 10.0, 1.0, format="%.2f")
            params['kernel'] = st.selectbox("Kernel", ["linear", "rbf", "poly"])

    # Train button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Train Model"):
        target = st.session_state.get("target_column")
        if not target:
            st.error("Target column not set. Please select in Import Data.")
            return show_import_data()
        df_proc = df.copy()

        # Apply missing handling
        for col in missing_drop:
            df_proc = df_proc.dropna(subset=[col])
        for col, (strat, fill) in missing_impute.items():
            imputer = SimpleImputer(strategy=strat, fill_value=fill)
            df_proc[[col]] = imputer.fit_transform(df_proc[[col]])

        # Split
        X = df_proc.drop(columns=[target])
        y = df_proc[target]

        # Build transformers
        transformers = []
        if scale_cols or any(col not in scale_cols for col in numeric_cols):
            num_steps = [("imputer", SimpleImputer(strategy="mean"))]
            if scale_cols:
                num_steps.append(("scaler", StandardScaler()))
            transformers.append(("num", Pipeline(num_steps), numeric_cols))
        if encode_cols:
            oh_cols = [c for c,m in encode_cols.items() if m=="OneHot"]
            ord_cols = [c for c,m in encode_cols.items() if m=="Ordinal"]
            if oh_cols:
                transformers.append(("onehot", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown='ignore'))]), oh_cols))
            if ord_cols:
                transformers.append(("ordinal", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ord", OrdinalEncoder())]), ord_cols))

        preprocessor = ColumnTransformer(transformers)

        # Instantiate model
        if model_choice == "RandomForest":
            model = RandomForestClassifier(**params)
        elif model_choice == "GradientBoosting":
            model = GradientBoostingClassifier(**params)
        elif model_choice == "SVM":
            model = SVC(**params)
        elif model_choice == "RandomForestRegressor":
            model = RandomForestRegressor(**params)
        elif model_choice == "GradientBoostingRegressor":
            model = GradientBoostingRegressor(**params)
        else:
            model = SVR(**params)

        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        metric = "R^2" if task_type == "Regression" else "Accuracy"
        st.success(f"Model trained! Test {metric}: {score:.4f}")
