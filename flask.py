from flask import Flask, request, jsonify
import pandas as pd
import json
import pickle
import os

app = Flask(__name__)

# Load the ML model
MODEL_PATH ='/Users/alexisalmeida/Documents/01. Imperial/01. Courses/Sensing IOT/Mood/ML/random_forest_experiment1.pkl'  # Replace with your model's path
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

LOG_FILE_PATH = "temp_log.json"
with open(LOG_FILE_PATH, "w") as log_file:
    log_file.write("[]")  # Initialize as an empty JSON array

# Register cleanup function to delete the log file on app exit
def cleanup_log_file():
    if os.path.exists(LOG_FILE_PATH):
        os.remove(LOG_FILE_PATH)




# GSR to EDA Conversion
def gsr_to_eda(gsr_adc_value, adc_max=4096, v_max=3.3, r_ref=165000):
    v_out = (gsr_adc_value / adc_max) * v_max
    if v_out <= 0 or v_out >= v_max:
        return None  # Invalid reading
    r_skin = r_ref * ((v_max - v_out) / v_out)
    if r_skin <= 0:
        return None  # Invalid reading
    return (1.0 / r_skin) * 1e6




@app.route("/receive_data", methods=["POST"])
def receive_data():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    # Extract sensor values
    hr = data.get("HR")
    st = data.get("ST")
    gsr = data.get("GSR")

    if hr is None or st is None or gsr is None:
        return jsonify({"error": "Incomplete data received"}), 400

    # Convert GSR to EDA
    eda = gsr_to_eda(gsr)

    if eda is None:
        return jsonify({"error": "Invalid GSR value"}), 400

    # Prepare data for ML model prediction
    new_data = pd.DataFrame({"HR": [hr], "ST": [st], "EDA": [eda]})

    # Make predictions using the ML model
    try:
        predictions = model.predict(new_data)
        prediction_probabilities = (
            model.predict_proba(new_data) if hasattr(model, "predict_proba") else None
        )

        response = {
            "HR": hr,
            "ST": st,
        #    "GSR": gsr,
            "EDA": eda,
            "predictions": predictions.tolist(),
            "prediction_probabilities": (
                prediction_probabilities.tolist()
                if prediction_probabilities is not None
                else None
            ),
        }
        try:
            with open(LOG_FILE_PATH, "r+") as log_file:
                log_contents = json.load(log_file)  # Read existing log
                log_contents.append(response)  # Add new entry
                log_file.seek(0)  # Move to start of file
                json.dump(log_contents, log_file, indent=4)  # Write updated log
                log_file.truncate()  # Remove any excess
        except Exception as e:
            print(f"Error writing to log file: {e}")
            return jsonify({"error": f"Logging failed: {str(e)}"}), 500
        return jsonify("OK"), 200

    except Exception as e:
        print(f"Error making predictions: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
