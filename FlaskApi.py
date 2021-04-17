# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 14:59:43 2021

@author: mnjaa
"""
'''import the required libraries'''

from flask import Flask,request
import pandas as pd
import numpy as np
import pickle



'''the flask app starting point'''
app=Flask(__name__) 


'''reading the file created in jupyter'''
pickle_in=open('classifier.pkl', 'rb')
classifier=pickle.load(pickle_in)

'''provide a decorate to provide a root path/ welcome page'''
@app.route('/')
def welcome():
    return "welcome All"

'''create a function, predict note authentication'''

@app.route('/predict')
def predict_note_authentication():
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    return "The predicted value is " + str(prediction)


@app.route('/predict_file', methods=["POST"])
def predict_note_file():
    df_test=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df_test)
    return "The predicted values for the csv are " + str(list(prediction))


if __name__=='__main__':
    app.run()