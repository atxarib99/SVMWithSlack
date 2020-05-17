"""
    This class will serve as an example class.
    An example perceptron.data file is included in the example folder that will be used.
"""

#import the SVM class to use
import SVM

#create a SVM class instance. We provided the filename as well as where the labels are located
svm = SVM.SVM("example/perceptron.data", labelPosition=4)

#optionally we can also do this like below
# svm = SVM.SVM()
# svm.import_data("example/perceptron.data", labelPosition=4)

#if we have a custom data model we want to import then...
# import examplemodeler as model
# svm = SVM.SVM()
# X,Y = model.import_data()
#the svm class will handle partitioning
# svm.loadTrainingData(X, Y)

#if we want to change the way the data is split into train,test,valid
#even split
# svm = SVM.SVM(partitioner=[1,1,1])
#also even split
# svm = SVM.SVM(partitioner=[10,10,10])
#50% train, 25% test, 25% valid
# svm = SVM.SVM(partitioner=[50,25,25])

#now we can train
svm.train()

#lets load some data that has no labels to predict on
#the exampletrainer just loads in a copy of perceptron.data with no labels
#here you would import your labeless data
import examplemodeler as model
X = model.importNoLabelData()
Y = svm.predict(X)
#print Y
for prediction in Y:
    print(prediction)
