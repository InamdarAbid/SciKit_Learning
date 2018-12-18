import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

#Load digits dataset
digits = datasets.load_digits() 

#Load SVM as clf
clf = svm.SVC(gamma=0.001, C=100)

#Divide data for training and testing
X,y = digits.data[:-10], digits.target[:-10]

#Fitting the data (Training)
clf.fit(X,y)

#Testing
print(clf.predict(digits.data[-5:-4]))
plt.imshow(digits.images[-5], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()