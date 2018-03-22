##########################################
###### Supervised Machine Learning #######
######           Algorithm         #######
##########################################
######           RESULTS           ############
###############################################
#Training Set Mean Absolute Error: 37722.8095 #
#Test Set Mean Absolute Error: 57030.2695     #
###############################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib

# Load the Data set
df = pd.read_csv("houseDataSet.csv")

#Remove the fields from the data set that don't want to include in the model
del df['house_number']
del df['unit_number']
del df['street_name']
del df['zip_code']


#Replace Categorical data with on-hot encoded data
features_df = pd.get_dummies(df, columns = ['garage_type', 'city'])

#remove the sale price from the feature data
del features_df['sale_price']

#Create the x and y arrays
X = features_df.as_matrix()
y= df['sale_price'].as_matrix()

# splitting data sets in a training set(70%) and a test set (30%)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

model = ensemble.GradientBoostingRegressor(
    n_estimators= 3000,    #How many Desistion trees to built (higher the model more Accurate it is but longer it takes to build)
    learning_rate = 0.05,   #Controls how much each additional decision tree influences the overall predicition(lower rates lead to higher accuracy but only if n_estimators set high )
    max_depth=9,           # Controls how many layers deep each individual decision tree can be
    min_samples_leaf=11,    # how many times a value must appear in the decision tree for it to make a decision baseed on it
    max_features=0.1,      # Precentage of features in the model that we randomly choose to consider each time a branch is created in the decision tree
    loss ='huber',         #Calculates the models error rate or cost as it learns
    random_state = 0       #Set to 0 to grante random states
)
model.fit(X_train, Y_train)

#save the trained model to a file so we can use it in other programs
joblib.dump(model, 'TrainedEstimator.pkl')

# Find the error rate on the training set using the best parameters
mse = mean_absolute_error(Y_train, model.predict(X_train))
print("Training Set Mean Absolute Error: %.4f" % mse)

# Find the error rate on the test set using the best parameters
mse = mean_absolute_error(Y_test, model.predict(X_test))
print("Test Set Mean Absolute Error: %.4f" % mse)




