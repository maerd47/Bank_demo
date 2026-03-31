import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score



# Create model instance
model = XGBClassifier()

# Load saved model
model.load_model("model/xgboost_model.json")

st.title("Bank Marketing Campaign Prediction")

input = st.text_input("Enter the path to the test data (CSV file):", "bank_data/validation.csv")

if st.button("Evaluate Model"):


    data_test = pd.read_csv(input)
    test = data_test.drop(columns = ['y'])
    test_label = data_test['y']


    # Make predictions on the test data
    y_pred = model.predict(test)

    # Evaluate the model
    accuracy = accuracy_score(test_label, y_pred)
    print(f"Accuracy: {accuracy*100:.2f}%")
    

    st.write(f"Accuracy: {accuracy*100:.2f}%")



