# ======================================================
# STREAMLIT APP FOR MUSHROOM CLASSIFICATION
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb


# -----------------------------
# Load Data
# -----------------------------

@st.cache_data
def load_data():
    data = pd.read_csv("mushrooms.csv")
    return data


df = load_data()


# -----------------------------
# Preprocessing
# -----------------------------

X = df.drop("class", axis=1)
y = df["class"]

encoders = {}

for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# -----------------------------
# Sidebar Model Selection
# -----------------------------

st.sidebar.title("Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose a model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)


# -----------------------------
# Model Initialization
# -----------------------------

def get_model(name):

    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1200)

    elif name == "Decision Tree":
        return DecisionTreeClassifier()

    elif name == "KNN":
        return KNeighborsClassifier(n_neighbors=7)

    elif name == "Naive Bayes":
        return GaussianNB()

    elif name == "Random Forest":
        return RandomForestClassifier(n_estimators=200)

    elif name == "XGBoost":
        return xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss"
        )


model = get_model(model_choice)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))


# -----------------------------
# UI Layout
# -----------------------------

st.title("Mushroom Classification App")
st.write("Predict whether a mushroom is edible or poisonous.")

st.write(f"### Selected Model: {model_choice}")
st.write(f"Model Accuracy: {round(accuracy,4)}")


# -----------------------------
# User Input
# -----------------------------

st.write("### Enter Mushroom Features")

user_input = {}

for feature in X.columns:
    options = list(encoders[feature].classes_)
    selected = st.selectbox(feature, options)
    user_input[feature] = encoders[feature].transform([selected])[0]


input_df = pd.DataFrame([user_input])

prediction = model.predict(input_df)[0]
result = target_encoder.inverse_transform([prediction])[0]

st.write("### Prediction Result:")
st.success(f"This mushroom is predicted as: **{result.upper()}**")
