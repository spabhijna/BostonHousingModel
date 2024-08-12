from flask import Flask, jsonify, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('/Volumes/Code/BostonHousingModel/linear_regression_model.pkl')

@app.route(rule='/')
def home():
    return render_template('index.html')

@app.route(rule='/predict', methods = ['POST'])
def predict():
    # Captures the features from the form
    features = [float(x) for x in request.form.values() ]
    # Convert them into DataFrame
    df = pd.DataFrame([features], columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
    # Predict the value
    prediction = model.predict(df)

    # Return the predicted value
    return render_template('index.html', prediction_text=f'Predicted House Price: ${prediction[0]:.2f}')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5001,debug=1)

