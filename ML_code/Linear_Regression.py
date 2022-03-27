"""
Created on Thu Mar 24 17:28:26 2022
@author: CSEMN
"""

import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('student_scores.csv')

#info about dataset
dataset.shape
dataset.head()
dataset.describe()

#Visualize data
dataset.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

#Preparing the Data
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Training
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Making Predictions
y_pred = regressor.predict(x_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

#Evaluation 
from sklearn.metrics import mean_absolute_error,mean_squared_error
from math import sqrt
print('Mean Absolute Error    :', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error     :', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', sqrt(mean_squared_error(y_test, y_pred)))

#Bonus: Visualize data
plt.scatter(x_train, y_train,c='r')
B0=regressor.intercept_
B1=regressor.coef_
#Y=B0+B1*X
plt.plot(x_train,B0+B1*x_train,c='b')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()