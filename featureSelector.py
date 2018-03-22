##########################################
######SHOWS DATA FEATURES THAT HAVE#######
######THE BIGGEST EFFECTS ONE THE  #######
######     ON THE ALGORTHEM        #######
##########################################
####         city_Martinezfort - 0.00%
#### city_Julieberg - 0.00%
#### city_New Michele - 0.00%
#### city_New Robinton - 0.00%
#### city_Rickytown - 0.00%
#### city_Davidtown - 0.03%
#### city_West Terrence - 0.06%
#### city_Port Daniel - 0.07%
#### city_Lake Jennifer - 0.07%
#### city_Jenniferberg - 0.09%
#### city_Leahview - 0.09%
#### city_West Brittanyview - 0.10%
#### city_West Gerald - 0.10%
#### city_Fosterberg - 0.10%
#### city_Amystad - 0.10%
#### city_Clarkberg - 0.11%
#### city_Toddshire - 0.11%
#### city_Wendybury - 0.12%
#### city_East Justin - 0.12%
#### city_Brownport - 0.13%
#### city_Joshuafurt - 0.14%
#### city_West Lydia - 0.15%
#### city_Lake Carolyn - 0.15%
#### city_East Janiceville - 0.15%
#### city_Scottberg - 0.17%
#### city_Lake Christinaport - 0.17%
#### city_Port Adamtown - 0.18%
#### city_Justinport - 0.19%
#### city_West Gregoryview - 0.19%
#### city_Port Jonathanborough - 0.20%
#### city_East Lucas - 0.20%
#### city_South Stevenfurt - 0.21%
#### city_Davidfort - 0.22%
#### city_Morrisport - 0.23%
#### city_Hallfort - 0.23%
#### city_East Amychester - 0.24%
#### city_Richardport - 0.26%
#### city_Lake Dariusborough - 0.30%
#### city_Lewishaven - 0.31%
#### city_North Erinville - 0.33%
#### city_Jeffreyhaven - 0.34%
#### city_South Anthony - 0.40%
#### city_Port Andrealand - 0.44%
#### city_West Ann - 0.50%
#### city_Chadstad - 0.53%
#### has_central_heating - 0.54%
#### city_Coletown - 0.56%
#### has_central_cooling - 0.56%
#### garage_type_detached - 0.61%
#### city_Lake Jack - 0.62%
#### garage_type_none - 0.73%
#### garage_type_attached - 0.78%
#### has_fireplace - 1.25%
#### stories - 1.37%
#### has_pool - 1.41%
#### half_bathrooms - 1.86%
#### full_bathrooms - 2.94%
#### num_bedrooms - 3.98%
#### carport_sqft - 4.10%
#### year_built - 14.77%
#### garage_sqft - 14.95%
#### livable_sqft - 20.54%
#### total_sqft - 20.61%
#########################################
import numpy as np
from sklearn.externals import joblib

#list of all features in the Data
feature_labelsNoCitys = np.array(['year_built', 'stories', 'num_bedrooms', 'full_bathrooms', 'half_bathrooms', 'livable_sqft', 'total_sqft', 'garage_sqft', 'carport_sqft', 'has_fireplace', 'has_pool', 'has_central_heating', 'has_central_cooling', 'garage_type_attached', 'garage_type_detached', 'garage_type_none', 'city_Amystad', 'city_Brownport', 'city_Chadstad', 'city_Clarkberg', 'city_Coletown', 'city_Davidfort', 'city_Davidtown', 'city_East Amychester', 'city_East Janiceville', 'city_East Justin', 'city_East Lucas', 'city_Fosterberg', 'city_Hallfort', 'city_Jeffreyhaven', 'city_Jenniferberg', 'city_Joshuafurt', 'city_Julieberg', 'city_Justinport', 'city_Lake Carolyn', 'city_Lake Christinaport', 'city_Lake Dariusborough', 'city_Lake Jack', 'city_Lake Jennifer', 'city_Leahview', 'city_Lewishaven', 'city_Martinezfort', 'city_Morrisport', 'city_New Michele', 'city_New Robinton', 'city_North Erinville', 'city_Port Adamtown', 'city_Port Andrealand', 'city_Port Daniel', 'city_Port Jonathanborough', 'city_Richardport', 'city_Rickytown', 'city_Scottberg', 'city_South Anthony', 'city_South Stevenfurt', 'city_Toddshire', 'city_Wendybury', 'city_West Ann', 'city_West Brittanyview', 'city_West Gerald', 'city_West Gregoryview', 'city_West Lydia', 'city_West Terrence'])

#Load the trained model created
model = joblib.load('TrainedEstimator.pkl')

# Create a numpy array based on the model's feature importance
importance = model.feature_importances_

# Sort the feature labels based on the feature importance rankings from the model
feature_indexes_by_importance = importance.argsort()

# Print each feature label, from most important to least important (reverse order)
for index in feature_indexes_by_importance:
    print("{} - {:.2f}%".format(feature_labelsNoCitys[index], (importance[index] * 100.0)))

