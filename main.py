import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Trying to unpickle estimator")
from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('LinearRegressionModelForCarData.pkl', 'rb'))  # Load your trained model
car = pd.read_csv('this_clean_car_data.csv')  # Load your cleaned car data CSV

# Ensure all columns are in the same order as used during model training
columns = ['car_name', 'kms_driven', 'fuel_type', 'transmission', 'ownership', 'manufacture', 'engine', 'Seats']


@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['car_name'].unique())
    year = sorted(car['manufacture'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()
    transmission = car['transmission'].unique()
    ownership = car['ownership'].unique()
    engine = car['engine'].unique()
    Seats = car['Seats'].unique()
    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, years=year, fuel_type=fuel_type,
                           transmission=transmission, ownership=ownership, engine=engine, Seats=Seats)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # Retrieve data from the form
    company = request.form.get('company')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    ownership = request.form.get('ownership')
    transmission = request.form.get('transmission')
    engine = request.form.get('engine')
    Seats = request.form.get('Seats')
    driven = request.form.get('kilo_driven')

    # Check for empty fields
    if not all([company, year, fuel_type, ownership, transmission, engine, Seats, driven]):
        return "Error: Please fill out all the fields."

    # Convert to proper data types
    try:
        year = int(year)
        driven = int(driven)
    except ValueError:
        return "Error: 'Year' and 'Kilometers driven' must be numeric."

    # Prepare data for prediction
    data = pd.DataFrame(data=[[company, driven, fuel_type, transmission, ownership, year, engine, Seats]],
                        columns=columns)

    # Make prediction
    prediction = model.predict(data)
    return str(np.round(prediction[0], 5))

if __name__ == '__main__':
    app.run(debug=True)