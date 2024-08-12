import pandas as pd
import joblib

def load_model(model_path):
    """Load and return the model from the specified path."""
    return joblib.load(model_path)

def get_user_input():
    """Get user input for prediction features."""
    print("Enter the features for prediction:")
    features = {
        'CRIM': float(input("CRIM: ")),
        'ZN': float(input("ZN: ")),
        'INDUS': float(input("INDUS: ")),
        'CHAS': float(input("CHAS: ")),
        'NOX': float(input("NOX: ")),
        'RM': float(input("RM: ")),
        'AGE': float(input("AGE: ")),
        'DIS': float(input("DIS: ")),
        'RAD': int(input("RAD: ")),
        'TAX': int(input("TAX: ")),
        'PTRATIO': float(input("PTRATIO: ")),
        'B': float(input("B: ")),
        'LSTAT': float(input("LSTAT: "))
    }
    return pd.DataFrame([features])

def make_prediction(model, input_df):
    """Make a prediction using the provided model and input DataFrame."""
    prediction = model.predict(input_df)
    return prediction[0]

if __name__ == "__main__":
    model_path = model_path = '/Volumes/Code/BostonHousingModel/linear_regression_model.pkl'
    model = load_model(model_path)

    # Get user input and make prediction
    input_df = get_user_input()
    predicted_value = make_prediction(model, input_df)
    print(f"Predicted MEDV: ${predicted_value:.2f}")
