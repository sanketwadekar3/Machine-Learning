from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def DecisionTree():
	iris = load_iris()
	
	data = iris.data
	target = iris.target
	
	data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.5, random_state = 0)
	
	classifier = tree.DecisionTreeClassifier()
	
	classifier.fit(data_train, target_train)
	
	predictions = classifier.predict(data_test)
	
	accuracy = accuracy_score(target_test, predictions)
	
	return accuracy
	
def KNeighbor():
	iris = load_iris()
	
	data = iris.data
	target = iris.target
	
	data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.5, random_state = 0)
	
	classifier = KNeighborsClassifier()
	
	classifier.fit(data_train, target_train)
	
	predictions = classifier.predict(data_test)
	
	accuracy = accuracy_score(target_test, predictions)
	
	return accuracy
	
def main():
	accuracy = DecisionTree()
	print("Accuracy of Decision Tree Classifier is ",accuracy*100,"%")
	
	accuracy = KNeighbor()
	print("Accuracy of K Neighbor Classifier is ",accuracy*100,"%")
	
if __name__ == '__main__':
	main()