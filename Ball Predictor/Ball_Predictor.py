from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd

def Ball_Predictor(weight,surface):
	
	dataset = pd.read_csv('Balls.csv')
	
	print(dataset)
	
	label_encoder = preprocessing.LabelEncoder()
	
	dataset['Pattern'] = label_encoder.fit_transform(dataset['Pattern'])
	dataset['Label'] = label_encoder.fit_transform(dataset['Label'])
	
	print(dataset['Pattern'].unique())
	print(dataset['Label'].unique())
	
	X = dataset[['Weight','Pattern']]
	Y = dataset[['Label']]
	
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
	
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X,Y)
	
	result = clf.predict([[weight,surface]])
	
	if result == 1:
		print("Your object looks like Tennis ball")
	elif result == 0:
		print("Your object looks like Cricket ball")
		
def main():
	print("------ Ball Predictor Case Study ------")
	
	weight = input('Enter weight of object')
	
	surface = input("What is the surface type of your object : Rough or Smooth ?")
	
	if surface.lower() == "rough":
		surface = 1
	elif surface.lower() == "smooth":
		surface = 0
	else:
		print("Error : Wrong input")
		exit()
	
	Ball_Predictor(weight,surface)
	
if __name__ == "__main__":
	main()