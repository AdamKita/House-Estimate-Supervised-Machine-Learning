##########################################
######TESTS ESTIMATOR PROGRAM WITH  ######
######         VALUE INPUTS         ######
##########################################
import pandas as pd
import csv
from sklearn.externals import joblib

#Load the model trained
model = joblib.load('TrainedEstimator.pkl')
dataArray= []
with open('RandomHouseData.csv') as csvfile:
    reader = pd.read_csv(csvfile, delimiter = ';')
    for row in reader:
        p = model.predict(row)
        print("This house has an estimated value of ${:,.2f}".format(p))
        ####CONVERT EACH VALUE TO FLOAT OR INT OR BOOL#####