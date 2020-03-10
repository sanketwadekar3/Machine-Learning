import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def HeadBrainPredictor():
	
	#Load data
	data = pd.read_csv('HeadBrain.csv')
	
	print("Size of the dataset : ",data.shape)
	
	X = data['Head Size(cm^3)'].values
	Y = data['Brain Weight(grams)'].values
	
	#Least Square method
	mean_x = np.mean(X)
	mean_y = np.mean(Y)
	
	n = len(X)
	
	numerator = 0
	denominator = 0
	
	#Equation of line is y = mx + c
	
	for i in range(n):
		numerator += (X[i] - mean_x)*(Y[i] - mean_y)
		denominator += (X[i] - mean_x)**2
		
	m = numerator/denominator
	
	c = mean_y - (m * mean_x)
	
	print("Slope of Regression line is : ",m)
	print("Y intercept of Regression line is : ",c)
	
	max_x = np.max(X)+100
	min_x = np.min(X)-100
	
	#Display plotting of above points
	x = np.linspace(min_x,max_x)
	
	y = m * x + c
	
	plt.plot(x, y, color='#58b970', label='Regression Line')
	
	plt.scatter(X, Y, color='#ef5423', label='Scatter Plot')
	
	plt.xlabel('Head Size in cm3')
	
	plt.ylabel('Brain weight in grams')
	
	plt.legend()
	plt.show()
	
	#Find goodness of fit i.e. R square
	ss_t = 0
	ss_r = 0
	
	for i in range(n):
		y_pred = m * X[i] + c
		ss_t += (Y[i] - mean_y)**2
		ss_r += (Y[i] - y_pred)**2
		
	r2 = 1 - (ss_r/ss_t)
	
	print(r2)
	
def main():
	print("User Defined Linear Regression on Head Brain dataset")
	HeadBrainPredictor()
	
if __name__ == '__main__':
	main()