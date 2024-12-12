import streamlit as st
import pandas as pd
import pickle

# Load your trained model
MODEL_PATH = "random_forest_experiment1.pkl"
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Streamlit app
st.title("Anxiety Prediction")

st.write(
    """
    This app predicts your anxiety level based on physiological parameters:
    - **HR (Heart Rate)**
    - **ST (Skin Temperature)**
    - **GSR (Galvanic Skin Response)**
    """
)

# Input fields for user data
hr = st.number_input("Enter your Heart Rate (HR):", min_value=0.0, step=1.0)
st_temp = st.number_input("Enter your Skin Temperature (ST):", min_value=0.0, step=0.1)
gsr = st.number_input("Enter your Galvanic Skin Response (GSR):", min_value=0.0, step=1.0)

# When the user clicks "Predict"
if st.button("Predict Anxiety Level"):
    if hr and st_temp and gsr:
        # Create a DataFrame for model input
        new_data = pd.DataFrame({"HR": [hr], "ST": [st_temp], "EDA": [gsr]})
        
        # Get prediction
        prediction = model.predict(new_data)[0]
        probabilities = model.predict_proba(new_data)

        st.write(f"### Predicted Anxiety Level: {prediction}")
        st.write(f"### Prediction Probabilities: {probabilities}")
    else:
        st.warning("Please enter all values to get a prediction.")
