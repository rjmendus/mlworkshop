from sklearn.datasets import load_iris
import numpy as np
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
#print(iris.feature_names)
#print(iris.target_names)
#print(iris.data[0])
#print(iris.target[0])

#Keeping the 0th, 50th and 100th row as test data
X = [0, 50, 100]

# Deleting the above mentioned rows from dataset using numpy (The new list will be training data)

xtrain = np.delete(iris.data, X, axis=0)
ytrain = np.delete(iris.target, X)

#The testing data is taken separately
xtest = iris.data[X]
ytest = iris.target[X]

#Training the data using decision tree
clf = DecisionTreeClassifier()
clf.fit(xtrain, ytrain)

#Predicting the data
print(ytest)
print("Prediction = ",clf.predict(xtest))
