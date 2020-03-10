import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def KNNaccuracy(weather, temp, n):
	dataset = pd.read_csv('PlayPredictor.csv')

	label_encoder = preprocessing.LabelEncoder()

	dataset['Weather'] = label_encoder.fit_transform(dataset['Weather'])
	dataset['Weather'].unique()
	
	dataset['Temperature'] = label_encoder.fit_transform(dataset['Temperature'])
	dataset['Temperature'].unique()

	dataset['Play'] = label_encoder.fit_transform(dataset['Play'])
	dataset['Play'].unique()
	
	data = dataset[['Weather','Temperature']]
	target = dataset[['Play']]

	data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.5)

	classifier = KNeighborsClassifier(n_neighbors = n)
	classifier.fit(data_train, target_train.values.ravel())
	predictions = classifier.predict(data_test)
	accuracy = accuracy_score(target_test, predictions)
	
	result = classifier.predict([[weather,temp]])
	
	if result == 0:
		print("It seems like you can't play")
		
	elif result == 1:
		print("It seems like you can play")
		
	print("Accuracy is: ",accuracy*100)
	
def PredictInput(n):
	weather = input(("Enter Weather: (Sunny, Overcast, Rainy)"))
	temp = input(("Enter Temperature: (Hot, Mild, Cool)"))
	
	if weather.lower() == "sunny":
		weather = 2
	elif weather.lower() == "overcast":
		weather = 0
	elif weather.lower() == "rainy":
		weather = 1
	else:
		print("Invalid input")
		exit()
		
	if temp.lower() == "hot":
		temp = 1
	elif temp.lower() == "mild":
		temp = 2
	elif temp.lower() == "cool":
		temp = 0
	else:
		print("Invalid input")
		exit()
	
	KNNaccuracy(weather, temp, n)
	
def main():
	n = int(input("Enter value of K: "))
	PredictInput(n)
	
if __name__ == '__main__':
	main()