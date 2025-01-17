import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # Importing Plotly Express as 'px'

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.figure_factory as ff


def home_page():
    st.header("Import File")
    # File uploader widget
    uploaded_file = st.file_uploader("Choose a file (CSV, Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Display the file name
        st.write(f"File uploaded: {uploaded_file.name}")
        
        # Read and display the file content
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        
        # Store the uploaded file in session state if needed
        st.session_state.uploaded_file = uploaded_file

        # Display the DataFrame
        st.write(df)
        st.dataframe(df.head())

            
def EDA():
    options = ["iris_example.csv"]  # default dataset
    if "all_dataset" in st.session_state:
        options.append("Uploaded dataset")

    # Create a selectbox to choose the dataset
    option = st.selectbox(
        "Select dataset:",
        options
    )
    
    st.write("You selected:", option)
    df = pd.read_csv(f"{option}")
    st.write(option)
    st.dataframe(df.head())
    # Display basic dataset info
    st.subheader("Basic Information")
    st.write(f"Shape of the dataset: {df.shape}")
    st.write(f"Columns in the dataset: {df.columns}")
    st.write(f"Missing values per column:\n{df.isnull().sum()}")
    
    # Display descriptive statistics
    st.subheader("Descriptive Statistics")
    st.write(df.describe())
    
    # Histograms for numerical columns
    st.subheader("Histograms for Numerical Features")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        st.write(f"Histogram for {col}")
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    correlation_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    
    # Boxplot for detecting outliers
    st.subheader("Boxplots for Outlier Detection")
    for col in num_cols:
        st.write(f"Boxplot for {col}")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        st.pyplot(fig)
    
    # Pairplot for feature relationships (if dataset is small)
    st.subheader("Pairplot of Features")
    if len(df) <= 500:  # Avoiding too large pairplot
        st.write("Pairplot (first 500 rows)")
        sns.pairplot(df)
        st.pyplot()
    else:
        st.write("Dataset too large for pairplot. Displaying first 500 rows pairplot.")
        sns.pairplot(df.head(500))
        st.pyplot()
    
    # Plotting a feature distribution with Plotly for interactivity
    st.subheader("Interactive Feature Distribution")
    feature = st.selectbox("Select feature for interactive plot", num_cols)
    if feature:
        fig = px.histogram(df, x=feature, nbins=20, title=f"Distribution of {feature}")
        st.plotly_chart(fig)

def contact_page():
    data = pd.read_csv("iris_example.csv")

    st.title("Exploratory Data Analysis (EDA) Dashboard")

    # Sidebar for plot selection
    plot_option = st.sidebar.selectbox("Choose a plot type", ["Histogram", "Boxplot", "Scatter Plot", "Heatmap"])

    # Sidebar for conditions based on the plot
    condition_column = None
    hue = None
    x_axis = None
    y_axis = None
    bins = None

    if plot_option == "Histogram":
        condition_column = st.sidebar.selectbox("Select column for histogram", data.select_dtypes(include=['float64', 'int64']).columns)
        bins = st.sidebar.slider("Select number of bins", min_value=5, max_value=50, value=20)

    elif plot_option == "Boxplot":
        condition_column = st.sidebar.selectbox("Select column for boxplot", data.select_dtypes(include=['float64', 'int64']).columns)
        hue = st.sidebar.selectbox("Select hue for boxplot", ["None"] + list(data.select_dtypes(include=['object']).columns))

    elif plot_option == "Scatter Plot":
        x_axis = st.sidebar.selectbox("Select X-axis", data.select_dtypes(include=['float64', 'int64']).columns)
        y_axis = st.sidebar.selectbox("Select Y-axis", data.select_dtypes(include=['float64', 'int64']).columns)
        hue = st.sidebar.selectbox("Select hue for scatter plot", ["None"] + list(data.select_dtypes(include=['object']).columns))

    elif plot_option == "Heatmap":
        corr_matrix = data.corr()  # Calculate correlation matrix
        heatmap_column = st.sidebar.selectbox("Select column for heatmap", corr_matrix.columns)

    # Button to generate the plot
    generate_button = st.sidebar.button("Generate Plot")

    if generate_button:
        # Plotting based on the selected options
        if plot_option == "Histogram":
            fig, ax = plt.subplots()
            ax.hist(data[condition_column], bins=bins)
            ax.set_title(f"Histogram of {condition_column}")
            st.pyplot(fig)

        elif plot_option == "Boxplot":
            fig, ax = plt.subplots()
            if hue == "None":
                sns.boxplot(x=data[condition_column], ax=ax)
            else:
                sns.boxplot(x=data[condition_column], hue=data[hue], ax=ax)
            ax.set_title(f"Boxplot of {condition_column}")
            st.pyplot(fig)

        elif plot_option == "Scatter Plot":
            fig, ax = plt.subplots()
            if hue == "None":
                sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)
            else:
                sns.scatterplot(x=data[x_axis], y=data[y_axis], hue=data[hue], ax=ax)
            ax.set_title(f"Scatter Plot of {x_axis} vs {y_axis}")
            st.pyplot(fig)

        elif plot_option == "Heatmap":
            fig, ax = plt.subplots()
            sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

def train():

# Load a dataset (e.g., Iris dataset)
    data = pd.read_csv("iris_example.csv")
    
    X = data.drop('species', axis=1)  # Features
    y = data['species']  # Target

        # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set up the Streamlit page for model selection and training
    st.title("Machine Learning Model Training and Evaluation")

    # Sidebar for model selection
    model_option = st.sidebar.selectbox("Choose a model", ["Random Forest", "Logistic Regression", "Support Vector Machine (SVM)"])

    # Sidebar for hyperparameter tuning based on model choice
    params = {}

    if model_option == "Random Forest":
        n_estimators = st.sidebar.slider("Number of trees", 10, 200, 100)
        max_depth = st.sidebar.slider("Max depth", 1, 20, 10)
        params = {"n_estimators": n_estimators, "max_depth": max_depth}

    elif model_option == "Logistic Regression":
        C = st.sidebar.slider("Regularization strength (C)", 0.01, 10.0, 1.0)
        solver = st.sidebar.selectbox("Solver", ['lbfgs', 'liblinear', 'saga'])
        params = {"C": C, "solver": solver}

    elif model_option == "Support Vector Machine (SVM)":
        C = st.sidebar.slider("Regularization parameter (C)", 0.1, 10.0, 1.0)
        kernel = st.sidebar.selectbox("Kernel", ['linear', 'poly', 'rbf', 'sigmoid'])
        params = {"C": C, "kernel": kernel}

    # Button to train the model
    train_button = st.sidebar.button("Train Model")

    # If the button is pressed, train the model
    if train_button:
        # Train the model based on the selected model and parameters
        if model_option == "Random Forest":
            model = RandomForestClassifier(**params)
        elif model_option == "Logistic Regression":
            model = LogisticRegression(**params)
        elif model_option == "Support Vector Machine (SVM)":
            model = SVC(**params)

        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Metrics: Accuracy, Classification Report, Confusion Matrix
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Display accuracy
        st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")

        # Display classification report
        st.write(f"### Classification Report:")
        st.text(class_report)

        # Display confusion matrix as a heatmap
        st.write("### Confusion Matrix:")
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

        # Visualize the model's decision boundaries (if possible, depending on the model)
        if model_option in ["Random Forest", "Logistic Regression"]:
            st.write("### Model Decision Boundaries (Only for 2D datasets)")

            # Reduce features to 2D for visualization (only for 2D problems like Iris dataset)
            if X_train.shape[1] == 2:
                # Create a mesh grid for plotting decision boundaries
                x_min, x_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
                y_min, y_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                    np.arange(y_min, y_max, 0.1))
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                # Plot decision boundaries
                plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.Paired)
                plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.Paired)
                plt.title(f"Decision Boundaries of {model_option}")
                plt.xlabel(X_train.columns[0])
                plt.ylabel(X_train.columns[1])
                st.pyplot()

    # Additional visualization using Plotly (e.g., ROC Curve, etc.) can also be added here if needed.
# Create sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["Home", "About", "Contact"])

# Conditional rendering based on the page selected
if page == "Home":
    home_page()
elif page == "About":
    EDA()
elif page == "Contact":
    contact_page()