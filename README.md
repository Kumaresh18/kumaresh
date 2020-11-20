# TASK 1
# Importing required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
%matplotlib inline

# Importing dataset
data=pd.read_csv('C:\\Users\\ELCOT\Desktop\scores.csv')
data.head()

# Info from the data
data.info()

# Statistical measures
data.describe()
# Plots
sns.set_style("whitegrid")
sns.scatterplot(x = 'Hours',y="Scores",data=data,color="orange")

x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
                             test_size=0.2, random_state=0)

# Building Linear regression model
lr=LinearRegression()

# Training model
lr.fit(x_train, y_train)

m=lr.coef_
c=lr.intercept_

line=m*x+c

plt.scatter(x,y,color="blue")
plt.plot(x, line, color="black")
plt.xlabel("Hours")
plt.ylabel("Scores")

# Prediciting the scores
y_pred=lr.predict(x_test)

# Actual VS predicted
data1=pd.DataFrame({"Actual":y_test, "Predicted":y_pred})
data1

# Predict the scores for 9 hr/day
hrs=[[9]]
prediction=lr.predict(hrs)
print("Predicted Score={}".format(prediction[0]))

# Mean Absolute Error
from sklearn import metrics
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
