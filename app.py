
import pickle

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__,  template_folder='templates')
pipe = pickle.load(open('pipe.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    args = request.form
    print(args)
    new = pd.DataFrame({
        'Location': [args.get('Location')],
        'Kilometers_Driven': [args.get('Kilometers_Driven')],
        'Fuel_Type': [args.get('Fuel_Type')],
        'Transmission': [args.get('Transmission')],
        'Owner_Type': [args.get('Owner_Type')],
        'Mileage': [args.get('Mileage')],
        'Engine': [args.get('Engine')],
        'Power': [args.get('Power')],
        'Seats': [int(args.get('Seats'))],
        'Brand': [args.get('Brand')],
        'Age': [int(args.get('Age'))]

    })

    prediction = round(float(pipe.predict(new)[0]), 2)
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    HOST = '127.0.0.1'
    PORT = 4000
    app.run(HOST, PORT)
