"""
Created on Thu Mar 24 22:46:30 2022
@author: csemn
"""
import pandas as pd
dataset = pd.read_csv("diabetes.csv")

#Preparing the Data
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=0)

#Training
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train)

#Making Predictions
y_pred = knn.predict(x_test)

#Evaluation 
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print('Confusion Matrix :- \n',cnf_matrix)

print("Accuracy :",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall   :",metrics.recall_score(y_test, y_pred))
print("F1 Score :",metrics.f1_score(y_test, y_pred))