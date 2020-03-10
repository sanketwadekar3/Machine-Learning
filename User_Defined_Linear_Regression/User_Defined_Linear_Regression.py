import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def LRPredictor():

	#Load data
	X = [1,2,3,4,5]
	Y = [3,4,2,4,5]
	
	print("Values of Independent variable X : ",X)
	print("Values of Dependent variable Y : ",Y)
	
	#Least Square method
	mean_x = np.mean(X)
	mean_y = np.mean(Y)
	
	print("Mean of Independent variable X : ",mean_x)
	print("Mean of Dependent variable Y : ",mean_y)
	
	n = len(X)
	
	numerator = 0
	denominator = 0
	
	#Equation of Line is y = mx + c
	
	for i in range(n):
		numerator += (X[i]-mean_x)*(Y[i]-mean_y)
		denominator += (X[i]-mean_x)**2
		
	m = numerator/denominator
	
	# c = y' - mx'
	
	c = mean_y - (m*mean_x)
	
	print("Slope of Regression line is : ",m)		# 0.4
	print("Y intercept of Regression line is : ",c)	# 2.4
	
	#Display plotting of above points
	x = np.linspace(1,6,n)
	
	y = m*x + c
	
	plt.plot(x,y,color='#58b970', label='Regression Line')
	plt.scatter(X,Y,color='#ef5423', label='Scatter Plot')
	
	plt.xlabel('X - Independent Variable')
	plt.ylabel('Y - Dependent Variable')
	
	plt.legend()
	plt.show()
	
	#Findout goodness of fit i.e. R square
	ss_t = 0
	ss_r = 0
	
	for i in range(n):
		y_pred = c + m*X[i]
		ss_t += (Y[i] - mean_y)**2
		ss_r += (Y[i] - y_pred)**2
	
	r2 = 1 - (ss_r/ss_t)
	
	print("Goodness of fit using R2 method is : ",r2)
	
def main():
	print("User Defined Linear Regression")
	LRPredictor()

if __name__ == '__main__':
	main()