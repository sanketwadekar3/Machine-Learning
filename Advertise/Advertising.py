import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def advertising():
	data = pd.read_csv('Advertising.csv')
	
	X = data[['TV','radio','newspaper']].values
	Y = data['sales'].values
	
	xtrain, xtest, ytrain, ytest = train_test_split(X,Y,test_size=0.5)
	
	reg = LinearRegression()
	
	reg.fit(xtrain,ytrain)
	
	pred = reg.predict(xtest)
	
	df = pd.DataFrame({'Actual': ytest, 'Predicted': pred})
	print(df)
	
def main():
	advertising()
	
if __name__ == '__main__':
	main()