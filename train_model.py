import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
print("Loading dataset...")
df = pd.read_csv("creditcard_2023.csv")

# Drop unnecessary columns
print("Preprocessing data...")
df.drop(columns=['id'], inplace=True)

# Scale the 'Amount' column
scaler = MinMaxScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Reduce dataset size for faster training (use 10% of data for testing purposes)
df_sample = df.sample(frac=0.1, random_state=42)

# Split data into features and target variable
X = df_sample.drop(columns=['Class'])
y = df_sample['Class']

# Split into training and testing sets
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Random Forest model with optimizations
print("Training model...")
rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Model training complete!")
print("Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Save the trained model and scaler
joblib.dump(rf_model, "rf_fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Streamlit UI
st.title("Credit Card Fraud Detection System")
st.write("Enter transaction details to check for fraud risk.")

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
if st.button("Check Fraud Risk"):
    prediction = rf_model.predict(input_data)[0]
    probability = rf_model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.error(f"🚨 Fraud Detected! Risk Score: {probability:.2f}")
    else:
        st.success(f"✅ Legitimate Transaction. Risk Score: {probability:.2f}")
