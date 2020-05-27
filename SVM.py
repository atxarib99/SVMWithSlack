"""
    A dry SVM with slack algorithm.
	Basic setup that needs a model to be imported to learn on
"""
import math
import numpy as np
from cvxopt import solvers, matrix
solvers.options['show_progress'] = False

class SVM:
	"""
		The SVM class that can be initialized to be used for training.
	"""
	w = None
	b = None
	ksi = None
	c = None
	C_list = []
	partitioner = [1,1,1]

	X = []
	Y = []

	def __init__(self, filename="", c=[.001, .01, .1, 1, 10, 1e+2, 1e+3, 1e+4, 1e+5], labelPosition=0, partitioner=[1,1,1]):
		"""
			Constructor for the SVM class.

			Parameters:
				filename: if a filename is provided it automatically imports the data in there
				c: A list of c values to try. c determines how much penalty to apply for mistakes
				labelPosition: which index the label is located in the data in filename
				partitioner: how to partition the data
			
			Result:
				An instance of the SVM class.
		"""
		
		super().__init__()
		if filename != "":
			self.import_data(filename, labelPosition=labelPosition)
		
		self.C_list = c
		self.partitioner = partitioner

	def import_data(self, filename, labelPosition=0):
		"""
		This import data method expects a csv dataset containing parameters and labels.
		It also expects all parameters to be numbers as they will be parsed into floats.
		If the dataset is not applicable to this, create a custom data modeler and load data in.
		
		Parameters:
			filename: the name of the file to open
			labelPosition: index of where the label will be located
			c: the list of xi parameters to attempt. These are constants that penalize getting an entry incorrect
		
		Result:
			Saves the imported data within the class to be used in training.
		"""

		#import csv since we are going to use it
		import csv
		#holds parameters and their associated labels
		Y = []
		X = []
		#open file
		with open(filename) as f:
			#read file
			reader = csv.reader(f, delimiter=",", quotechar="\"")
			#for each row
			for row in reader:
				#add label
				Y.append(row[labelPosition])
				#holds parameters
				newList = []
				#for remaining items in row
				for i in range(0, len(row)):
					if i == labelPosition:
						continue
					#add parameter to list
					newList.append(float(row[i]))
				#add list to X
				X.append(newList)

		# self.X = np.array(X, dtype='float')
		# self.Y = np.array(Y, dtype='float')
		self.X = X
		self.Y = Y

			
	def slack_svm(self,X,Y,c):
		"""
		Runs SVM with slack using cvxopt's solver
		Transforms data into something cvxopt can read and returns result
		If a model cannot be build it returns None for all three outputs.

		Parameters:
			X: The data paremeters
			Y: The associated labels
			c: xi. the penalty constant for mislabeling.
		
		Result:
			Weights: Weights of each parameter
			Bias: The bias or offset
			ksi:
		"""

		# finds the features length
		n_feature = len(X[:,0])
		# gets sample size
		n_sample = Y.size
		n_paras = n_feature + 1 + n_sample
		# construct P
		P = np.zeros(n_paras)
		for i in range(n_feature):
			P[i]=1
		P = np.diag(P)

		# construct q
		q = np.zeros(n_paras)
		for i in range(n_sample):
			q[n_feature+1+i]=c

		# construct G phase 1, consider y(wx+b)>=1-ksi
		G = []
		for i in range(n_sample):
			# form: y_i*x_i,y_i,0..0,1,0..0
			tmp = np.zeros(n_paras)
			x_i = X[:,i]
			y_i = Y[i]
			tmp[0:n_feature] = y_i*x_i
			tmp[n_feature] = y_i
			tmp[n_feature+1+i] = 1
			G.append(tmp)

		# construct G phase 2, consider ksi >= 0
		for i in range(n_sample):
			tmp = np.zeros(n_paras)
			tmp[n_feature+1+i] =1
			G.append(tmp)
		G = np.array(G)

		# construct h
		h=np.zeros(n_sample*2)
		for i in range(n_sample):
			h[i]=1

		# transform Gx >= h to Gx <= h
		G=-G;h=-h
		ret = solvers.qp(matrix(P),matrix(q),matrix(G),matrix(h))
		solution = ret['x']

		# decompose solution to w,b,ksi
		w = solution[0:n_feature]
		w = np.array(w).reshape(n_feature,1)
		b = solution[n_feature]
		ksi = list(solution[n_feature+1:])
		good = self.verify(X,Y,w,b,ksi)
		if good:
			return w,b,ksi
		else:
			return None, None, None

	def F(self,w,b,x):
		""" 
		Function definition of how to compute label. Simply f(x) = wx+b
		
		Parameters:
			w: weights
			b: bias
			x: dataset to apply to
		
		Results:
			y: generates list of predictions based on function definition. matches len(x) 
			
		"""

		return np.dot(w.T,x)+b

	def get_accuracy(self,Y,func):
		""" 
		Calculates the accuracy by comparing the function to a known label set Y

		Parameters:
			Y: Known label set for dataset used in func
			func: function that generated a label set
		
		Results:
			accuracy: the accuracy of the func on the known label set. 1-100%. 0-0%
		"""

		R = np.multiply(Y.T,func)
		n_right = len(R[R>0])
		accuracy = float(n_right)/len(Y)
		return accuracy

	def verify(self,X,Y,w,b,ksi):
		""" checks the correctness of result parameter """

		n_sample = len(Y)
		for i in range(n_sample):
			y_i = Y[i]
			x_i = X[:,i]
			if y_i*(np.dot(w.T,x_i)+b) + ksi[i] < 1:
				print("ERROR FIND !")
				return 0
		print("Result PASS!")
		return 1

	def partitionData(self,X,Y):
		""" Partitions the dataset of X entries with Y labels into 3 seperate unique datasets for train, test, validation

			Parameters:
				X: The entries that contain the parameter for the prediction
				Y: The associated label for the parameters X

			Result:
				X_train = Entries for training
				Y_train = Associated labels for training set
				X_test  = Entries for testing
				Y_test  = Associated labels for test set
				X_valid = Entries for validation
				Y_valid = Associated labels for validation set
		"""

		#validate partitioner
		if len(self.partitioner) != 3:
			raise PartitioningException("The partitioner was invalid.")

		#holds size of data
		size = len(X)

		#calculate partition amounts
		trainSize = self.partitioner[0] / sum(self.partitioner) * size
		testSize  = self.partitioner[1] / sum(self.partitioner) * size + trainSize
		validSize = self.partitioner[2] / sum(self.partitioner) * size + testSize

		#cast sizes to int
		trainSize = int(trainSize)
		testSize  = int(testSize)
		validSize = int(validSize)
		print(trainSize, testSize, validSize)
		
		#holds train,valid,test sets
		X_train = []
		Y_train = []
		X_test  = []
		Y_test  = []
		X_valid = []
		Y_valid = []

		#populate training set
		for i in range(0, trainSize):
			X_train.append(X[i])
			Y_train.append(Y[i])
		#populate test set
		for i in range(trainSize, testSize):
			X_test.append(X[i])
			Y_test.append(Y[i])
		#populate validation set
		for i in range(testSize, validSize):
			X_valid.append(X[i])
			Y_valid.append(Y[i])
		
		return np.array(X_train, dtype='float').T, np.array(Y_train, dtype='float').T, np.array(X_test, dtype='float').T, np.array(Y_test, dtype='float').T, np.array(X_valid, dtype='float').T, np.array(Y_valid, dtype='float').T

	def predict(self, X, postiveValue=1, negativeValue=-1):
		"""
			Takes an input array X and creates predictions for it using trained weights.

			Parameters:
				X: An array of parameters
				positiveValue: what value to add to Y if prediction is a yes (typically 1)
				negativeValue: what value to add to Y if prediction is a no (typically -1 or 0)
			
			Result:
				Y: An array of predicted labels
		"""
		
		#if weights haven't been trained yet, throw an exception
		if self.w is None or self.b is None or self.c is None:
			raise NoTrainedWeights("Weights have not been trained yet. Try svm.train()")

		#create numpy array for provided X
		npX = np.array(X, dtype='float').T

		#create predictor 
		predictor = self.F(self.w, self.b, npX)

		#holds predictions
		Y = []
		#output predictions
		for prediction in predictor[0]:
			if prediction < 0:
				Y.append(negativeValue)
			else:
				Y.append(postiveValue)
		
		#return predictions
		return Y
			
	def train(self):
		X_t,Y_t,X_test,Y_test,X_valid,Y_valid = self.partitionData(self.X,self.Y)
		accuracy = {'valid':[], 'train':[]}
		
		for c in self.C_list:
			w,b,ksi = self.slack_svm(X_t,Y_t,1)
			predictor = self.F(w,b,X_valid)
			accuracy['valid'].append(self.get_accuracy(Y_valid,predictor))
			predictor = self.F(w,b,X_t)
			accuracy['train'].append(self.get_accuracy(Y_t,predictor))

		tmp = accuracy['valid']
		# find best parameter combination
		max_accuracy = max(tmp)
		max_configs = list(filter(lambda x:x[1]==max_accuracy, zip(self.C_list,tmp) ))

		print("\nTrain accuracy", accuracy['train'])
		print("\nValidation accuracy",accuracy['valid'])
		c_accuracy = {}
		# verify best para on test set
		for c,acc in max_configs:
			w,b,ksi = self.slack_svm(X_t,Y_t,c)
			predictor = self.F(w,b,X_test)
			accuracy = self.get_accuracy(Y_test,predictor)
			c_accuracy[c] = (w,b,ksi,accuracy)
		
		#find best solutions
		max_accuracy = 0
		cw = None
		cb = None
		cksi = None
		matchingc = self.C_list[0]
		for item in c_accuracy.keys():
			if c_accuracy[item][3] > max_accuracy:
				cw,cb,cksi,max_accuracy = c_accuracy[item]
				matchingc = item
		
		self.w = cw
		self.b = cb
		self.ksi = cksi
		self.c = matchingc
		
		print("Accuracy on test set", max_accuracy)
		print("Ready to predict! use svm.predict(X) to get prediction array Y")

	def loadTrainingData(self,X,Y):
		""" Loads prebuilt training data """

		# self.X = np.array(X)
		# self.Y = np.array(Y)
		self.X = X
		self.Y = Y

	def appendData(self, X, Y):
		""" Appends more data onto the current training data. This may be useful if your data is already split. """
		
		for item in X:
			self.X.append(item)
		for item in Y:
			self.Y.append(item)
	
	def setCList(self, c):
		"""
			Sets the list of Cs to try. c determines how much penalty to apply for mistakes

			Parameters:
				c: the list of Cs
		"""
		self.C_list = c
	
	def setPartitioner(self, partitioner):
		"""
			Sets the partitioner. The partitioner decides how to split up the data

			Parameters:
				partitioner: a list of length three that defines how to partition the data into train,test,valid respectively
		"""

		#raise an error if not of length 3
		if len(partitioner) != 3:
			raise PartitioningException("Partitioner must be of length 3. Train,Test,Valid")

		self.partitioner = partitioner


class ImproperFormatException(Exception):
	""" An exception that lets the user knows that the given dataset could not be automatically parsed """
	pass

class PartitioningException(Exception):
	""" An exception that lets the user know the partition they entered was not valid. """
	pass

class NoTrainedWeights(Exception):
	""" An exception that lets the user know that the model has not been trained on yet. """
	pass

