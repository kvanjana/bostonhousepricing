import pickle
from django.shortcuts import render
from flask import Flask, request, app, jsonify, url_for, render_template
import pandas as pd
import numpy as np

##(__name__) is the starting point of the application from where it will run
app = Flask(__name__)
##load the model
with (open('linear_regression.pkl','rb')) as file1:
   model = pickle.load(file1)
with (open('scaler.pkl','rb')) as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)
