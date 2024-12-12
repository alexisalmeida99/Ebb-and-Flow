import numpy as np
import pandas as pd
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, log_loss, make_scorer
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


filepath = "/Users/alexisalmeida/Documents/01. Imperial/01. Courses/Sensing IOT/Mood/Data/master_experiment_1.csv"
def load_and_preprocess_data(filepath):
    """
    Loads and preprocesses the data from a CSV file.

    Parameters:
    - filepath (str): Path to the CSV file.

    Returns:
    - X (DataFrame): Feature matrix.
    - y (Series): Target vector.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded '{filepath}'.")
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found in the current directory.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading the data: {e}")
        raise

    # Drop the 'Unnamed: 0' column if it exists
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
        print("Dropped 'Unnamed: 0' column.")

    # Rename columns for consistency
    df.columns = ["HR", "ST", "GSR", "Labels"]
    print("Renamed columns to ['HR', 'ST', 'GSR', 'Labels'].")

    # Check for missing values
    if df.isnull().sum().any():
        print("Warning: Missing values detected. Proceeding to drop missing values.")
        df = df.dropna()

    # Separate features and target
    X = df[["HR", "ST", "GSR"]]
    y = df["Labels"]

    return X, y


def perform_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Parameters:
    - X (DataFrame): Feature matrix.
    - y (Series): Target vector.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Controls the shuffling applied to the data before splitting.

    Returns:
    - X_train, X_test, y_train, y_test: Split datasets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Performed train-test split with test size = {test_size*100}%.")
    return X_train, X_test, y_train, y_test


def custom_scoring():
    """
    Defines custom scoring metrics for cross-validation.

    Returns:
    - scoring (dict): Dictionary of scoring metrics.
    """
    scoring = {
        "accuracy": "accuracy",
        "log_loss": "neg_log_loss",  # Negate because sklearn expects higher scores
    }
    return scoring


def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    """
    Trains the Random Forest model, performs cross-validation, evaluates on test set,
    and saves the trained model.

    Parameters:
    - X_train (DataFrame): Training feature matrix.
    - y_train (Series): Training target vector.
    - X_test (DataFrame): Testing feature matrix.
    - y_test (Series): Testing target vector.

    Returns:
    - None
    """
    # Initialize the Random Forest model with specified hyperparameters
    rf_model = RandomForestClassifier(
        n_estimators=400,
        min_samples_split=2,
        min_samples_leaf=1,
        max_depth=8,
        bootstrap=False,
        random_state=42,
    )
    print("Initialized Random Forest with specified hyperparameters.")

    # Create a pipeline with StandardScaler and the Random Forest model
    # Note: Random Forests do not require feature scaling, but included for consistency
    pipeline = make_pipeline(StandardScaler(), rf_model)
    print("Created a pipeline with StandardScaler and Random Forest.")

    # Define cross-validation parameters
    cv_folds = 10
    scoring = custom_scoring()

    print(f"Starting {cv_folds}-fold cross-validation...")
    start_time = time.time()

    # Perform cross-validation
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv_folds,
        scoring=scoring,
        return_train_score=False,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Cross-validation completed in {elapsed_time:.2f} seconds.")

    # Calculate mean and standard deviation for metrics
    mean_accuracy = np.mean(cv_results["test_accuracy"]) * 100
    std_accuracy = np.std(cv_results["test_accuracy"]) * 100
    # Negate back to positive
    mean_log_loss = -np.mean(cv_results["test_log_loss"])
    # For percentage representation
    std_log_loss = np.std(cv_results["test_log_loss"]) * 100

    print(f"Cross-Validation Accuracy: {mean_accuracy:.2f}% (+/- {std_accuracy:.2f}%)")
    print(f"Cross-Validation Log Loss: {mean_log_loss:.4f} (+/- {std_log_loss:.4f})")

    # Train the model on the entire training set
    print("Training the model on the entire training set...")
    pipeline.fit(X_train, y_train)
    print("Model training completed.")

    # Evaluate on the test set
    print("Evaluating the model on the test set...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)

    test_accuracy = accuracy_score(y_test, y_pred) * 100
    test_log_loss = log_loss(y_test, y_pred_proba)

    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test Log Loss: {test_log_loss:.4f}")

    # Save the trained model
    model_filename = "random_forest_experiment1.pkl"
    try:
        with open(model_filename, "wb") as file:
            pickle.dump(pipeline, file)
        print(f"Trained model saved as '{model_filename}'.")
    except Exception as e:
        print(f"Error saving the model: {e}")


def main():
    # Define the path to your combined dataset
    data_filepath = "./data/master_experiment_1.csv"

    # Load and preprocess the data
    X, y = load_and_preprocess_data(data_filepath)

    # Perform train-test split
    X_train, X_test, y_train, y_test = perform_train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model, perform cross-validation, evaluate, and save
    train_and_evaluate_model(X_train, y_train, X_test, y_test)



if __name__ == "__main__":
    main()