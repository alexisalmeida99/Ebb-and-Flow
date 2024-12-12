# Ebb and Flow
This project contains the code for my project Ebb and Flow. 

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
Once these two applications are running, you will then need to flash your arduino with the sketch i've provided. This sketch will send a packet to your flask server

In order to run the training script you need to download the data from the https://github.com/rs2416/Detecting_Social_Anxiety datasets. 
Once you download the data, store it ina folder called Data in the same directory. In the train script, you'll notice I used master_experiment_1.csv. Make sure the dataset
you've downloaded corresponds to the filename in the train.py script.
