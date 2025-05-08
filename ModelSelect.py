import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
import streamlit as st
import os
import joblib

def show_modelselect():
    st.title("🔧 Data Preparation for Training")

    # Load dataset (from session state or upload)
    # get the uploaded or selected dataset path
    path = st.session_state.get("dataset_path", None)
    if not path or not os.path.exists(path):
        st.warning("Please upload or select a dataset on the Home / Import Data page first.")
        st.stop() 
    
    df = pd.read_csv(path)

    target = st.session_state.get("target_column")
    missing_drop = st.session_state.get("missing_drop", [])
    missing_impute = st.session_state.get("missing_impute", {})
    scale_cols = st.session_state.get("scale_columns", [])
    encode_cols = st.session_state.get("encode_columns", {})
    
    # Select task and model
    st.subheader("Task & Model Selection")
    task = st.selectbox("Task Type", ["Classification", "Regression"])

    if task == "Classification":
        model_name = st.selectbox("Model", ["RandomForest", "GradientBoosting", "SVM"])
    else:
        model_name = st.selectbox(
            "Model", ["RandomForestRegressor", "GradientBoostingRegressor", "SVR"]
        )

    # Hyperparameters
    st.subheader("Hyperparameters")
    params = {}
    if "RandomForest" in model_name:
        params["n_estimators"] = st.number_input("n_estimators", min_value=10, max_value=1000, value=100, step=10)
        params["max_depth"] = st.number_input("max_depth", min_value=1, max_value=100, value=5, step=1)
    elif "GradientBoosting" in model_name:
        params["n_estimators"] = st.number_input("n_estimators", min_value=10, max_value=1000, value=100, step=10)
        params["learning_rate"] = st.number_input("learning_rate", min_value=0.001, max_value=1.0, value=0.1, format="%.3f")
    else:
        params["C"] = st.number_input("C", min_value=0.01, max_value=10.0, value=1.0, format="%.2f")
        params["kernel"] = st.selectbox("Kernel", ["linear", "rbf", "poly"])

    # Train action
    if st.button("🚀 Train Model"):
        if not target:
            st.warning("Target column not set. Please select it on the Import Data page.")
            st.stop()

        if df.empty:
            st.warning("Dataframe not loaded. Please upload a dataset.")
            st.stop()
        
        # Prepare data copy
        df_proc = df.copy()

        # Drop rows with missing values in specified columns
        if missing_drop:
            df_proc.dropna(subset=missing_drop, inplace=True)

        # Impute missing values
        for col, (strategy, fill) in missing_impute.items():
            imputer = SimpleImputer(strategy=strategy, fill_value=fill)
            df_proc[col] = imputer.fit_transform(df_proc[[col]]).ravel()  

        # Split features/target
        X = df_proc.drop(columns=[target])
        y = df_proc[target]

        # Identify numeric and categorical columns
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Build transformers
        transformers = []
        
        # Numeric pipeline
        num_steps = [("imputer", SimpleImputer(strategy="mean"))]
        if scale_cols:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(num_steps), numeric_cols))

        # Categorical pipelines
        if encode_cols:
            oh_cols = [c for c, m in encode_cols.items() if m == "OneHot"]
            ord_cols = [c for c, m in encode_cols.items() if m == "Ordinal"]
            if oh_cols:
                transformers.append(
                    ("onehot", Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown='ignore'))
                    ]), oh_cols)
                )
            if ord_cols:
                transformers.append(
                    ("ordinal", Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder())
                    ]), ord_cols)
                )

        preprocessor = ColumnTransformer(transformers)

        # Instantiate model
        model_map = {
            "RandomForest": RandomForestClassifier,
            "GradientBoosting": GradientBoostingClassifier,
            "SVM": SVC,
            "RandomForestRegressor": RandomForestRegressor,
            "GradientBoostingRegressor": GradientBoostingRegressor,
            "SVR": SVR
        }
        ModelClass = model_map[model_name]
        model = ModelClass(**params)

        # Full pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # Train/test split and fit
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)

        # Display result
        metric_name = "Accuracy" if task == "Classification" else "R²"
        st.success(f"Model trained! Test {metric_name}: {score:.4f}")
        joblib.dump(pipeline, 'trained_pipeline.pkl')

        # Provide a download button for the user to download the trained model
        with open('trained_pipeline.pkl', 'rb') as f:
            st.download_button(
                label="📥 Download trained model",
                data=f,
                file_name="trained_pipeline.pkl",
                mime="application/octet-stream"
            )