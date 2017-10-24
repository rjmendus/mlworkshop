#To create a func to calculate Euclidean distance
from scipy.spatial import distance
def eucli(a,b):
	return distance.euclidean(a,b)

#Creating our own algorithm for knn classifier
class myKNN():
	def fit(self, X_train, Y_train):
		self.X_train = X_train
		self.Y_train = Y_train

	def predict(self, X_test):
		predictions = []
		for row in X_test:
			labels = self.closest(row)
			predictions.append(labels)
		return predictions
	def closest(self, row):
		best_dist = eucli(row, self.X_train[0])
		best_index = 0
		for i in range(1, len(self.X_train)):
			dist = eucli(row, self.X_train[i])
			if dist < best_dist:
				best_dist = dist
				best_index = i
		return self.Y_train[best_index]


#from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

iris = load_iris()
features = iris.data
labels = iris.target

#splitting data using sklearn (taining:test -> 70-30)
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=.3)

knn = myKNN()
knn.fit(X_train, Y_train)

p = knn.predict(X_test)

#checking accuracy
from sklearn.metrics import accuracy_score
print("Accuracy =",accuracy_score(Y_test, p))


