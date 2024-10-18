import numpy as np
import pandas as pd
from flask import Flask, request, render_template, flash
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for flash messages

# Load the trained model and scaler
try:
    model = joblib.load('logreg_model.joblib')
    scaler = joblib.load('scaler.joblib')
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Get input values from form
            open_price = float(request.form['open'])
            close_price = float(request.form['close'])
            low_price = float(request.form['low'])
            high_price = float(request.form['high'])
            is_quarter_end = int(request.form['is_quarter_end'])

            # Calculate features
            open_close = open_price - close_price
            low_high = low_price - high_price

            # Create a feature array and scale it
            features = np.array([[open_close, low_high, is_quarter_end]])
            scaled_features = scaler.transform(features)

            # Make prediction
            prediction = model.predict(scaled_features)
            probability = model.predict_proba(scaled_features)[0][1]

            # Prepare result
            result = "Up" if prediction[0] == 1 else "Down"
            confidence = f"{probability:.2%}" if prediction[0] == 1 else f"{1-probability:.2%}"

            return render_template('index.html', prediction=result, confidence=confidence,
                                   open_price=open_price, close_price=close_price,
                                   low_price=low_price, high_price=high_price,
                                   is_quarter_end=is_quarter_end)
        except KeyError as e:
            flash(f"Missing form field: {str(e)}", 'error')
        except ValueError as e:
            flash(f"Invalid input: {str(e)}", 'error')
        except Exception as e:
            flash(f"An error occurred: {str(e)}", 'error')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)