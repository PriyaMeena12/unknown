# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load("Student_mark_predictor_model.pkl")

df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global df

    input_features = [int(x) for x in request.form.values()]
    features_values = np.array(input_features)

    # Validate input hours
    if input_features[0] < 0 or input_features[0] > 24:
        return render_template('index.html', prediction_text='Please enter the valid hours between 1 to 24')

    output = model.predict([features_values])[0].round(2)

    # Add to DataFrame and save to CSV
    df = pd.concat([df, pd.DataFrame({'Study Hours': [input_features[0]], 'Predicted Output': [output]})], ignore_index=True)
    print(df)
    df.to_csv('smp_data_from_app.csv', index=False)

    return render_template('index.html', prediction_text=f'You will get [{output}%] marks when you study [{input_features[0]}] hours per day.')

if __name__ == "__main__":
    app.run()
