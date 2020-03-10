import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def KNNaccuracy(n):
	dataset = pd.read_csv('WinePredictor.csv')
	data = dataset.iloc[:,1:13]
	target = dataset[['Class']]
	
	data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.3)
	
	sc = StandardScaler()
	data_train = sc.fit_transform(data_train)
	data_test = sc.transform(data_test)
	
	classifier = KNeighborsClassifier(n_neighbors = n)
	classifier.fit(data_train, target_train.values.ravel())
	
	predictions = classifier.predict(data_test)
	accuracy = accuracy_score(target_test, predictions)
	print("Accuracy of KNN: ",accuracy*100)
	
def main():
	n = int(input("Enter value of K: "))
	KNNaccuracy(n)
	
if __name__ == '__main__':
	main()