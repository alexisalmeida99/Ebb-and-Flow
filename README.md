# Ebb and Flow
This project contains the code for my project Ebb and Flow. Ebb & Flow is a wearable device designed to monitor and manage anxiety by providing real-time feedback on physiological signals such as heart rate, skin temperature, and galvanic skin response (GSR). This project leverages IoT technologies and machine learning to empower users with insights into their emotional states and tools to improve mental well-being.

Youtube Video Demo: https://youtu.be/e0VjgX1bkUU

**Acknowledgments**
This project was developed as part of the DESE71003 – Sensing and Internet of Things course at Imperial College London. Special thanks to Professor David Boyle for providing well-documented datasets that enabled the machine learning pipeline.

**Project Overview**
Stress and anxiety can hinder creativity and productivity. The Ebb & Flow device addresses this challenge by collecting and analyzing physiological data to predict anxiety levels.The system provides insights through a live web dashboard, allowing users to track their emotional states and monitor the impact of relaxation techniques, such as breathing exercises.

**Features**
Physiological Signal Monitoring: Measures heart rate, skin temperature, and GSR in real time.
Machine Learning Prediction: Utilizes a Random Forest Classifier to predict anxiety with an accuracy of 97.85%.
Live Data Visualization: Displays real-time predictions on a user-friendly   dashboard.
Local Data Storage: Ensures data privacy by storing all information locally and clearing the database after the session ends.

**System Architecture**
The project follows an end-to-end pipeline:

1. Sensors: The device uses the MAX30102 sensor (for heart rate and temperature) and the Grove GSR sensor to collect physiological data. 
2. Arduino Nano ESP32: Processes raw data and sends it to a local Flask server.
3. Flask Server: Preprocesses the data and runs it through the machine learning pipeline.
4. Streamlit Dashboard: Visualizes predictions and provides real-time feedback to the user.

**Installation and Setup**

To build this project, first create a virtual environment and install the required packages. This can be done with the following commands

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Once the packages are installed, you should be able to run the flask.py and streamlit.py applications.
To run flask, simply make sure the venv is activated and run the command
```bash
python flash.py
```
for streamlit, its
```bash
streamlit run streamlit.py
```
---
Once these two applications are running, you will then need to flash your arduino with the sketch I've provided. This sketch will send a packet to your flask server

In order to run the training script you need to download the data from the https://github.com/rs2416/Detecting_Social_Anxiety datasets. 
Once you download the data, store it in a folder called Data in the same directory. In the train script, you'll notice I used master_experiment_1.csv. Make sure the dataset
you've downloaded corresponds to the filename in the train.py script.



**Usage**

Built the circuit. 
Attach the sensors to the user. 
Start the Flask server to begin data collection and processing.
Open the Streamlit dashboard to view real-time predictions of anxiety levels.
Practice stress-reduction techniques and monitor changes in physiological data on the dashboard.


<img width="272" alt="image" src="https://github.com/user-attachments/assets/8b78f0d4-a302-4fa2-830d-2df1144aa6f5" />

