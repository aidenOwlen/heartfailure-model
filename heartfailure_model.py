# -*- coding: utf-8 -*-
"""
We will compare random forest to logistic regression in predicting health failure using heart_failure_clinical_records_dataset 
We will first build a logistic regression from scratch,
Fit the model in the logistic regression and random forest, and compare metrics

"""
import pickle
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import IsolationForest #contamination 0.1, n_estimators = 500
import pickle



class logistic_model:
	"""Class returned after building our logistic model
	parameters : 
	weights : Learned by the model
	Iterations : list with indices of iterations during training
	Loss : Computed loss on training set
	learning_rate : learning_rate used to train our model

	Methodes : 
	predict : take X_test as parameter return (m,1) np array of predictions
	Plot_train : Plot iterations vs loss for during the training of our model
	true positive : take X_test, Y_test and returns True positives for our model
	evaluate : take X_text, Y_test and returns metrics dictionary of our model ( specificity, sensibility, recall, precision, F1, accuracy)
	"""
	def __init__(self,weights,iterations,loss,learning_rate):
		self.iterations = iterations 
		self.loss = loss
		self.weights = weights
		self.learning_rate = learning_rate

	def predict(self,X_test):
		return np.round(sigmoid(X_test @ self.weights))

	def plot_train(self):
		plot_iterations_against_loss(self.iterations,self.loss)

	def true_positive(self,X_test,Y_test):
		 Y_pred = self.predict(X_test)
		 TP = Y_pred.T@Y_test
		 return TP
	
	def evaluate(self,X_test,Y_test):
		Y_pred = self.predict(X_test)
		TP = TN = FP = FN = 0
		for index in range(Y_pred.shape[0]):
			if Y_test[index][0] == Y_pred[index][0] == 1:
				TP+= 1
			elif Y_test[index][0] == Y_pred[index][0] == 0:
				TN += 1
			elif Y_test[index][0] == 1  and Y_pred[index][0] == 0:
				FN += 1
			else:
				FP += 1
		specificity = TN/(TN+FP)
		sensibility = recall = TP/(TP+FN)
		precision = TP/(TP + FP)
		F1 = 2*precision*recall/(precision + recall)
		accuracy = (TP + TN) / (TP + TN + FN + FP)
		return {"specificity":specificity, "sensibility":sensibility, "recall":recall,"precision":precision,"F1":F1,"accuracy":accuracy}
		

def sigmoid(z):
	""" Activate the hypothesis
	Take hypothesis as parameter and return the sigmoid function """
	return 1/(1+np.exp(-z))

def cost_function(hypothesis,Y):
	"""Compute the cost, parameters : 
	hypothesis of the model : shape ( mx1)
	Y : ground truth, shape (mx1) 
	returns the cost as as scalar"""
	m = hypothesis.shape[0]
	return (-1/m) * np.sum(Y * np.log(hypothesis) + ( 1 - Y)* np.log(1-hypothesis))

def plot_iterations_against_loss(X,Y):
	plt.plot(X,Y)
	plt.xlabel("Iterations")
	plt.ylabel("Loss")
	plt.legend()
	plt.show()


def prepare_data():
	"""Preparind data : 
	1) load data 
	2) Drop time column
	3) Normalize/standarize data
	4) split : X_train, X_test, Y_train, Y_test, Y is death event occurence, regardless of time interval
	5) Create a new column of ones ( for the bias )
	6) Create vector of weights
	7) Convert to numpy
	8) Return X_train,Y_train,X_test,Y_test,weights
	"""
	df = pd.read_csv("dataset/heart_failure_clinical_records_dataset.csv") #load data
	df.drop("time",axis=1,inplace=True) # Delete time column
	deaths = sum(df["DEATH_EVENT"] == 1) # I thought this could be useful in this case, to use a weighted sum loss function because of the class imbalance ( normal +++)
	alives = sum(df["DEATH_EVENT"] == 0)
	total = df.shape[0]
	weighted_loss = (alives/total, deaths/total) # Weighted sum
	
	column_to_standarize = ["age","creatinine_phosphokinase","platelets","ejection_fraction","serum_creatinine",
	"serum_sodium"] # Columns we are going to standarize

	for column in column_to_standarize: #Standarization
		df[column] = (df[column] - df[column].mean()) / df[column].std()
	
	data_split = train_test_split(df.drop("DEATH_EVENT",axis =1 ), df["DEATH_EVENT"], test_size=0.2, random_state=2) #Splitting data
	
	X_train,X_test, Y_train, Y_test = [_.to_numpy() for _ in data_split] # Converting to numpy
	X_train,X_test= np.c_[np.ones(X_train.shape[0]), X_train],np.c_[np.ones(X_test.shape[0]), X_test] #Adding ones for the bias weight
	Y_train,Y_test = Y_train.reshape(Y_train.shape[0],1), Y_test.reshape(Y_test.shape[0],1) #Reshaping (m,) to (m,1) to avoid linear algebra issues
	weights = np.zeros(X_train.shape[1]).reshape(X_train.shape[1],1) #Weight vector

	return X_train,Y_train,X_test,Y_test,weights


def logitic_regression(X_train,Y_train, weights,learning_rate, epochs):
	"""Training function, 
	Parameters :
	X_train : shape (m,n)
	Y_train : shape (m,1)
	weights : shape (n,1)
	learning_rate : float/int
	epochs : number of iterations 
	returns the trained model as a class
	"""
	iterations = []
	loss = []
	i = 0
	m = X_train.shape[0]
	while i < epochs:
		hypothesis = sigmoid(X_train @ weights)
		jacobian = (X_train.T @ (hypothesis-Y_train))/m
		weights = weights - (learning_rate * jacobian)
		iterations.append(i)
		loss.append(cost_function(hypothesis,Y_train,))
		i+= 1
	return logistic_model(weights,iterations,loss,learning_rate)



if __name__ == "__main__":
	X_train,Y_train,X_test,Y_test,weights = prepare_data() #Prepare data

	#Logistic model
	model_logistic = logitic_regression(X_train, Y_train, weights,learning_rate=0.01, epochs = 10000)
	#model_logistic.predict(X_test)
	#model_logistic.true_positive(X_test,Y_test)
	#metrics = model_logistic.evaluate(X_test,Y_test)
	#model_logistic.plot_train()

	#Random forest model
	model_forest = RandomForestClassifier(max_features=3,max_depth = 4,random_state = 1) #initiate
	model_forest.fit(X_train,Y_train) #Train
	forest_pred = model_forest.predict(X_test) #PRedict
	print(model_forest.score(X_test,Y_test))
	print(classification_report(Y_test, forest_pred))
	pickle.dump(model_forest, open("model_heart_failure.pkl","wb")) #Save random forest model
	print("done")


	