"""
	This file will serve to be an example trainer.
"""

def import_data():
	"""
		This method will import data from perceptron.data and return it as a tuple (X,Y) which are list of parameters and list of labels respectively.

		Result:
			(X,Y) a tuple where X is a list of parameters and Y is a list of labels 
	"""

	#import csv since we are about to use it
	import csv
	#open the file
	with open("example/perceptron.data") as f:
		#where X and Y will be stored
		X = []
		Y = []
		#create csv reader
		reader = csv.reader(f, delimiter=",", quotechar="\'")
		#for each row in data
		for row in reader:
			#temp list for this rows parameters
			tempList = []
			#we know first 4 are parameter objects, so add them to the temp list
			for i in range(0, 4):
				tempList.append(float(row[i]))
			#add this tempList into list of parameters
			X.append(tempList)
			#add label for this row which we know is at index 4
			Y.append(float(row[4]))
	#return the data
	return X,Y

def importNoLabelData():
	"""
		This method will import data from perceptronnolabel.data and return it as a tuple X which is list of parameters

		Result:
			X: X is a list of parameters
	"""

	#import csv since we are about to use it
	import csv
	#open the file
	with open("example/perceptronnolabel.data") as f:
		#where X will be stored
		X = []
		#create csv reader
		reader = csv.reader(f, delimiter=",", quotechar="\'")
		#for each row in data
		for row in reader:
			#temp list for this rows parameters
			tempList = []
			#we know first 4 are parameter objects, so add them to the temp list
			for i in range(0, 4):
				tempList.append(float(row[i]))
			#add this tempList into list of parameters
			X.append(tempList)
	#return the data
	return X