# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 20:55:27 2023

@author: AnS
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mentioning the datatypes of our inputs to the model
class model_input(BaseModel):
    bedrooms: int
    bathrooms: int
    sqft_living: int
    sqft_lot: int

# Loading the saved model
house_model = pickle.load(open('House_model.sav', 'rb'))

@app.post('/house_prediction')
def house_predd(input_parameters: model_input):
    try:
        # Extract the values from input_parameters
        bedrooms = input_parameters.bedrooms
        bathrooms = input_parameters.bathrooms
        sqft_living = input_parameters.sqft_living
        sqft_lot = input_parameters.sqft_lot

        # Create a 2D array for prediction (one sample)
        input_2d_array = [bedrooms, bathrooms, sqft_living, sqft_lot]

        # Predict with your Linear Regression model
        prediction = house_model.predict([input_2d_array])

        # Return the prediction
        return {'prediction': prediction[0]}
    except Exception as e:
        return {'error': str(e)}
