import streamlit as st
import pandas as pd
import pickle
from flask import Flask, request, jsonify
import threading
import plotly.express as px
import time
from scipy.ndimage import uniform_filter1d
import threading
from flask_cors import CORS
from threading import Lock
import requests
import json


LOG_FILE_PATH = "temp_log.json"
data_store = pd.DataFrame(columns=["HR", "ST", "EDA", "Prediction"])


def fetch_live_data():
    global data_store
    try:
        # Read the JSON log file
        with open(LOG_FILE_PATH, "r") as log_file:
            log_contents = json.load(log_file)

        # Convert JSON content to a DataFrame
        if isinstance(log_contents, list):
            new_data = pd.DataFrame(log_contents)
        else:
            st.error("Unexpected log file format.")
            return

        # Update the data store
        data_store = new_data

    except FileNotFoundError:
        st.error("Log file not found.")
    except Exception as e:
        st.error(f"Error fetching live data: {e}")


st.title("Ebb and Flow")
st.write("This dashboard shows real-time predictions of anxiety.")
# Main dashboard
st.header("Live Sensor Data")
button = st.button("Get Reading")

# Periodically fetch live data
if button:
   # st.write("Fetching...")
    # Use Streamlit's placeholder to dynamically update the content
    status_placeholder = st.empty()
   
    # Continuously fetch and process data until interrupted
    try:

        fetch_live_data()
     #   time.sleep(2)  # Simulate data fetching every 2 seconds
            
            # Get the latest prediction from data_store
        if not data_store.empty:
            last_row = data_store.iloc[-1]
            last_prediction = last_row['predictions'][0]
                
                # Update the status based on the last prediction
            if last_prediction:
                st.write("You're probably anxious")
            else:
                st.write('Flowing')


    except Exception as e:
        st.write(f"Fetching stopped manually: {e}")
try:
    st.write(data_store.describe())
except Exception as e:
    pass
#log_placeholder= st.dataframe(data_store.iloc[-1])