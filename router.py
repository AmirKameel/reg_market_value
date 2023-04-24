# Import the Libraries
import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request
import joblib
import os
# the function I craeted to process the data in utils.py
from utils import preprocess_new


# Intialize the Flask APP
app = Flask(__name__)

# Loading the Model
import pickle

#import class_def
#from sklearn.externals import joblib
import joblib

model = joblib.load('model_linreg.pkl')
    
#if __name__=='__main__':
   # with open('model_randfor.pkl', 'rb') as f:
    #    model = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')

# Route for Predict page


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':  # while prediction
        Goals = int(request.form['Goals'])
        Assist =int (request.form['Assist'])
        Pass = int(request.form['Pass'])
        PassCompleted = int(request.form['PassCompleted'])
        Tackle_Won = (request.form['Tackle_Won'])
        AerWon = int(request.form['AerWon'])

        Cross = int(request.form['Cross'])

        CrossCompleted = int(request.form['CrossCompleted'])
        AerLost = int(request.form['Aerlost'])

        # Remmber the Feature Engineering we did
        
        # Concatenate all Inputs
        X_new = pd.DataFrame({'G/90': [Goals], 'Assist': [Assist], 'Pass': [Pass], 'PassCompleted': [PassCompleted], 'Tackle_Won': [Tackle_Won],
                              'AerWon': [AerWon], 'Cross': [Cross], 'CrossCompleted': [CrossCompleted] , 'AerLost' : [AerLost]
                             
                              })

        # Call the Function and Preprocess the New Instances
        X_processed = preprocess_new(X_new)

        # call the Model and predict
        y_pred_new = model.predict(X_new)
        #y_pred_new = '{:.4f}'.format(y_pred_new[0])
        

        return render_template('predict.html', pred_val=y_pred_new)
    else:
        return render_template('predict.html')


# Route for About page
@app.route('/about')
def about():
    return render_template('about.html')


# Run the App from the Terminal
if __name__ == '__main__':
    app.run(debug=True)
