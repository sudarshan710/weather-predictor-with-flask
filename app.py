from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from dotenv import load_dotenv
import dropbox
import os

app = Flask(__name__)
load_dotenv()

dbx = dropbox.Dropbox(os.getenv('DP_KEY'))

def download_file(file_path, download_path):
    with open(download_path, 'wb') as f:
        metadata, res = dbx.files_download(path=file_path)
        f.write(res.content)

file_path = '/model_pkl/regressor_model.pkl' 

download_path = 'model.pkl'  
download_file(file_path, download_path)


model = joblib.load('model.pkl')
features = ['Year', 'Month', 'Day', 'Humidity', 'Wind Speed (km/h)', 'Visibility (km)', 'Pressure (millibars)']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    try:
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        humidity = float(request.form['humidity'])
        windspeed = float(request.form['windspeed'])
        visibility = float(request.form['visibility'])
        pressure = float(request.form['pressure'])

        inst = pd.DataFrame([[year, month, day, humidity, windspeed, visibility, pressure]], columns=features)
        result = model.predict(inst)

        return render_template('index.html', prediction=round(result[0], 3))

    except ValueError as e: 
        return render_template('index.html', error='Invalid Input.')


if __name__ == '__main__':
    app.run(debug=True)