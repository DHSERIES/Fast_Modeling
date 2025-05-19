import streamlit as st
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.title("Model Training and Evaluation")


# Load dataset (from session state or upload)
# get the uploaded or selected dataset path
path = st.session_state.get("dataset_path", None)
if not path or not os.path.exists(path):
    st.warning("Please upload or select a dataset on the Home / Import Data page first.")
    st.stop() 

df = pd.read_csv(path)

task = st.selectbox("Select Task", ["Classification", "Regression"])

# Sidebar - Target selection
target = st.selectbox("Select Target Column", df.columns)

# Features
X = df.drop(columns=[target])
y = df[target]

# Train-test split
test_size = st.slider("Test set size (%)", 10, 50, 20)
random_state = st.number_input("Random State", value=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=random_state
)

# Model selection
if task == "Classification":
    model_name = st.selectbox("Select Model", ["Logistic Regression", "Random Forest"])
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier()
else:
    model_name = st.selectbox("Select Model", ["Linear Regression", "Random Forest"])
    if model_name == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor()

# Train button
if st.button("Train Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write(f"## {task} Results using {model_name}")
    
    if task == "Classification":
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        st.write(f"**Accuracy:** {acc:.4f}")
        st.write(f"**F1 Score:** {f1:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        st.write("**Confusion Matrix:**")
        fig, ax = plt.subplots()
        ax.matshow(cm, cmap=plt.cm.Blues)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha='center', va='center')
        st.pyplot(fig)

        # ROC Curve for binary
        if len(pd.unique(y)) == 2:
            y_prob = model.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
            ax2.plot([0,1], [0,1], linestyle='--')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('ROC Curve')
            ax2.legend(loc='lower right')
            st.pyplot(fig2)

    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        st.write(f"**RMSE:** {rmse:.4f}")
        st.write(f"**R^2 Score:** {r2:.4f}")

        # Residual Plot
        residuals = y_test - y_pred
        fig, ax = plt.subplots()
        ax.scatter(y_pred, residuals)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        st.pyplot(fig)
