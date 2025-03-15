import pandas as pd
import numpy as np
import streamlit as st
import joblib
import hashlib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Load the trained model and scaler
rf_model = joblib.load("rf_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Set Streamlit theme
st.set_page_config(page_title="WON Credit Fraud Detection", page_icon="💳", layout="centered")

# Custom CSS for blue and white theme
st.markdown(
    """
    <style>
        body {
            background-color: #e6f2ff;
            color: #00274d;
        }
        .stButton > button {
            background-color: #004080 !important;
            color: white !important;
            border-radius: 5px;
            border: none;
            padding: 10px;
        }
        .stTextInput>div>div>input {
            border-radius: 5px !important;
            padding: 10px;
        }
        .stTitle {
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: #004080;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# User authentication system
CREDENTIALS_FILE = "users.json"

def load_credentials():
    if not os.path.exists(CREDENTIALS_FILE):
        return {}
    with open(CREDENTIALS_FILE, "r") as file:
        return json.load(file)

def save_credentials(credentials):
    with open(CREDENTIALS_FILE, "w") as file:
        json.dump(credentials, file)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def is_valid_password(password):
    import re
    return (
        len(password) >= 8 and
        any(char.isupper() for char in password) and
        any(char.islower() for char in password) and
        any(char.isdigit() for char in password) and
        any(char in "!@#$%^&*()-_=+[]{};:,.<>?/" for char in password)
    )

# Session state to persist login status
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

credentials = load_credentials()
st.markdown("<h1 class='stTitle'>WON Credit Fraud Detection Software</h1>", unsafe_allow_html=True)

if not st.session_state.logged_in:
    menu = st.sidebar.selectbox("Menu", ["Login", "Create Account"])
    
    if menu == "Create Account":
        st.subheader("Create an Account")
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        if st.button("Sign Up"):
            if new_username in credentials:
                st.error("Username already exists! Try another.")
            elif not is_valid_password(new_password):
                st.error("Password must contain at least 8 characters, an uppercase letter, a lowercase letter, a number, and a special character.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                credentials[new_username] = hash_password(new_password)
                save_credentials(credentials)
                st.success("Account created successfully! Please log in.")
                st.balloons()
    
    elif menu == "Login":
        st.subheader("Login to Your Account")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if username in credentials and credentials[username] == hash_password(password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome {username}! You are logged in.")
                st.rerun()
            else:
                st.error("Invalid Username or Password")
else:
    st.subheader(f"Welcome, {st.session_state.username}!")
    st.write("Enter transaction details to check for fraud detection.")

    # Explanation of V1-V28 features and Transaction Amount
    st.info("**V1-V28**: These are anonymized features representing transaction characteristics after PCA transformation. **Transaction Amount**: The actual value of the transaction.")

    # User input fields for transaction features
    features = {}
    for i in range(1, 29):
        features[f'V{i}'] = st.number_input(f'V{i}', value=0.0)
    
    features['Amount'] = st.number_input("Transaction Amount", value=0.0)

    # Convert input to DataFrame
    input_data = pd.DataFrame([features])

    # Scale the Amount column
    input_data['Amount'] = scaler.transform(input_data[['Amount']])

    # Make prediction
    if st.button("Check Fraud Status"):
        prediction = rf_model.predict(input_data)[0]
        
        if prediction == 1:
            st.error("🚨 ALERT! This transaction is **FRAUDULENT**. Immediate action required!")
        else:
            st.success("✅ This transaction is **LEGITIMATE**. No fraud detected.")
    
    # Display Fraud vs Non-Fraud Transactions Pie Chart
    st.subheader("Fraud vs. Non-Fraud Distribution")
    labels = ["Legitimate", "Fraudulent"]
    sizes = [80, 20]  # Example distribution
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures pie chart is circular
    st.pyplot(fig)
    
    # Feature Importance Bar Chart
    st.subheader("Feature Importance in Fraud Detection")
    feature_importance = rf_model.feature_importances_
    feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importance = feature_importance[sorted_indices]
    
    fig, ax = plt.subplots()
    sns.barplot(y=sorted_features[:10], x=sorted_importance[:10], ax=ax)
    ax.set_title("Top 10 Important Features")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    st.pyplot(fig)
