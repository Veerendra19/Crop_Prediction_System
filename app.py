from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('random_forest_crop_model.pkl')
scaler = joblib.load('scaler.pkl')

# Features for input
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Label encoder to decode the crop label
df = pd.read_csv('Crop_recommendation.csv')
le = LabelEncoder()
le.fit(df['label'])


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input from user
        user_input = [float(request.form[f]) for f in features]

        # Preprocess and predict
        input_scaled = scaler.transform([user_input])
        predicted_label = model.predict(input_scaled)

        # Decode the label
        recommended_crop = le.inverse_transform(predicted_label)[0]

        return render_template('index.html', recommended_crop=recommended_crop)

    return render_template('index.html', recommended_crop=None)


if __name__ == '__main__':
    app.run(debug=True)