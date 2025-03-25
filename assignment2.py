# Import pandas and our model
from sklearn.tree import DecisionTreeClassifier #a classifier is different from a gresser, because teh classifer makes decisions for discreet dependent variable
#regressor is for continous dependent variables
import pandas as pd
import numpy as np
import patsy as pt 

from sklearn.metrics import accuracy_score #the proportion of observations for whcih your able to make an accurate decision
from sklearn.model_selection import train_test_split #this will help us to break our data into different chunks 

#what we will use to train the data (data_meal)
data_meal = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")


#this the data we'll make predictions with (for now ignore) data_pred
data_pred = pd.read_csv(" https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")



#meal is the DEPENDENT VARIABLE

# As usual, Patsy makes data prep easier, use ptsy to make and x and y matrix
#<-this is how to do it with matrices: Y, X = pt.dmatrices("meal ~ -1 + Iced + Starbucks_DS_Vanilla  + Tea_Half_and_Half + Nuts_Cashews", data=data_meal) #-1 means no interncept term, meal is the depedent variable (y), everything else are our independent variables (x)


#probably have to remove the columns wuth ID and datetime in them? Since they arent applicable
Y = data_meal['meal'] #this meakes it so that the Depenedt variable (y) meal
X = data_meal.drop(['meal', 'id', 'DateTime'], axis =1) #makes (x) the dependent variable everything but meal which is Y, also drops DateTime and Id because I dont think those are there to help predicy anything but just help find whatever


#remember that capital X, Y are different from lowercase x, y (dont want them to get overwritten)
#x and y are training datat
#xt and yt are testing data
#will randomly shuffle observations into training and testing bins 
x, xt, y, yt = train_test_split(X, Y,  test_size=0.33, random_state=42) #train test split is going to reserve 1/3 or .333 of the data as test data so we 
# can test how well our model performs 
#randomstate is giving a starting spot for the random draws, allows us to make this repeatable, means that we wil get the same shuffles of data everytime we run this line, becasuse im forcing the random draws to start at the same spot everytime

#so column names in list
column_names = data_meal.columns.tolist()

#print(column_names)

print(X)

#now that we have the training and testing data we can build our model

#model is defined by creating an instance of teh decision tree classifier object 
model = DecisionTreeClassifier()

#now fit the model
res = model.fit(x,y)

modelFit = res


#now we can make predictions using fitted model (based on your x's it will predict what y's you should get )

#THIS IS THE IN-SAMPLE ACCURACY WHICH ISNT REALLY IMPORTANT

pred1 = modelFit.predict(x)
print("\n\nIn-sample accuracy: %s%%\n\n" 
 % str(round(100*accuracy_score(y, pred1), 2))) #the ys are then compared by the accuracy score against truth, and we see how often were right

 #INSTEAD WE WWANT OUT OF SAMPLE ACCURACY

#big difference between In and Out of sample. Our model is VERY overfit right now 

#to fix we are going to try to make the model less complex, model is good sepcifically for our data but not applicable 
#to all outcomes (so right now our model IS only good for this sepcific instance which is not good)

# we will reduce the model to just the generalizable characteritsitcs

pred_train = modelFit.predict(x)
pred_test = modelFit.predict(xt)

#in sample accuracey with training data
print("\n\nIn-sample accuracy: %s%%\n\n" 
 % str(round(100*accuracy_score(y, pred_train), 2))) #In-sample accuracy: 94.35%

#out of sample accuracy with testing data
print("\n\nOut-of-sample accuracy: %s%%\n\n" 
% str(round(100*accuracy_score(yt, pred_test), 2)))





#I think now i need to implement the actual test data data_pred

#apparently some of the columns in the test data set have nan values so i need to fiind and remove them?
data_pred.isnull().sum() #this shows me which variables have nans and how many they have


#it looks like all of the values for Meal in data_pred are Nan?? maybe i just need to remove them or somethin
data_pred.replace('nan', np.nan) #this replace the nan with the np.nan(NaN)

#this remove the NaN  data_pred.dropna() , #dont do this i dont think its working lol

#print(data_pred.dropna() )

#took all of thevalues in the meal call and made it empty, so no Nan
#need to make sure that these are empty because utimately this what we are trying to predict
#did they get a meal (1)
#did they not get a meal (0)
#based on what else they ordered 
data_pred['meal'] = ''

data_pred['meal'] = 0 #i just mad all the values in meal =0 because Idl it wasnt working with empty (said that the predictions had to atleast match the type of values)

y2 = data_pred['meal'] #this meakes it so that the Depenedt variable (y) meal
x2 = data_pred.drop(['meal', 'id', 'DateTime'], axis =1) #makes (x) the dependent variable everything but meal which is Y, also drops DateTime and Id because I dont think those are there to help predicy anything but just help find whatever

#after checking again we can see that all of the Nans are gone
data_pred.isnull().sum()

pred_almost = modelFit.predict(x2)

#now need to make sure that that values in pred are integers, when i chedked its saying that some of the numbers are being classfied as numpy.int64
pred = [int(x) if isinstance(x, (str, np.int64, np.int32)) else x for x in pred_almost]
#this helps to ensure that they are all integers 

print("\n\nTest (pred) data accuracy: %s%%\n\n" 
 % str(round(100*accuracy_score(y2, pred), 2))) #the ys are then compared by the accuracy score against truth, and we see how often were right

