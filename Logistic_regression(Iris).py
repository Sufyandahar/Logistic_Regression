# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# dataset = pd.read_csv('Iris.csv')
dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
dataset.head()

# slicing the dataset
X = dataset.iloc[:, :4].values
y = dataset.iloc[:, 4].values


# spliting the dataset into traning and testing s
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 5)

# KNN classifer
from sklearn.linear_model import LogisticRegression
classifier =LogisticRegression()

# fit the model
classifier.fit(X_train, y_train)

# prediction on test data
y_pred = classifier.predict(X_test)

################# confusion matrix ###################
from sklearn.metrics import confusion_matrix
print("Confusion matrix\n", confusion_matrix(y_test, y_pred))

# precision
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# f1-score
from sklearn.metrics import f1_score
print("f1-score (macro) : ",f1_score(y_test, y_pred, average='macro'))
print("f1-score (micro) : ", f1_score(y_test, y_pred, average='micro'))
# check accuracy
from sklearn.metrics import accuracy_score
print("Accuracy_score   :", accuracy_score(y_test, y_pred))

##########################################################  Final Prediction  ##############################################################

result = classifier.predict([[4.9,2.4,3.3,1]])
print("Final Prediction :",result)