##########################################
######SHOWS DATA SET TO HTML WEBPAGE######
##########################################
import pandas as pd
import webbrowser
import os

#reads the fata set into a table
dataTable = pd.read_csv("houseDataSet.csv")

#Create a table that views the first 100 cols of the Data set
html = dataTable[0:100].to_html()
with open ("DataSetToHTML.html","w") as f:
    f.write(html)

#open the web page in the browser
dataSetFile = os.path.abspath("DataSetToHTML.html")
webbrowser.open("file://{}".format(dataSetFile))
