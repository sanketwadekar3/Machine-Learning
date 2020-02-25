from scipy.spatial import distance
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def euc(a, b):
	return distance.euclidean(a, b)
	
class KNN():
	def fit(self, TrainingData, TrainingTarget):
		self.TrainingData = TrainingData
		self.TrainingTarget = TrainingTarget
		
	def predict(self, TestData):
		predictions = []
		for row in TestData:
			label = self.closest(row)
			predictions.append(label)
		return predictions
		
	def closest(self, row):
		best_distance = euc(row, self.TrainingData[0])
		best_index = 0
		for i in range(1,len(self.TrainingData)):
			dist = euc(row,self.TrainingData[i])
			if dist < best_distance:
				best_distance = dist
				best_index = i
		return self.TrainingTarget[best_index]
		
def KNeighbor():
	border = "-"*50
	
	iris = load_iris()
	
	data = iris.data
	target = iris.target
	
	data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.5, random_state = 0)

	classifier = KNN()
	
	classifier.fit(data_train,target_train)
	
	predictions = classifier.predict(data_test)
	
	accuracy = accuracy_score(target_test, predictions)
	
	return accuracy
	
def main():
	accuracy = KNeighbor()
	print("Accuracy of User Defined KNN is ",accuracy*100,"%")
	
if __name__ == "__main__":
	main()
