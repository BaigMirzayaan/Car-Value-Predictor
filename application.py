from flask import Flask, render_template, request, redirect
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
cors = CORS(app)

pipeline, metadata = joblib.load(open('LinearRegressionModel_meta_1.pkl', 'rb'))
car = pd.read_csv("Cleaned_car_data.csv")


@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    company_chosen = request.form.get('company')

    name_chosen = request.form.get('car_model')
    year_chosen = request.form.get('year')
    fuel_chosen = request.form.get('fuel_type')
    driven = request.form.get('kms_driven')

    input_data = pd.DataFrame([[name_chosen, company_chosen, year_chosen, driven, fuel_chosen]],
                              columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    print("Raw Input Data:")
    print(input_data)

    prediction = pipeline.predict(input_data)

    print("Prediction:")
    print(prediction)
    prediction[0]=prediction[0]/85

    return str(np.round(prediction[0], 2))


if __name__ == '__main__':
    app.run(debug=True)
