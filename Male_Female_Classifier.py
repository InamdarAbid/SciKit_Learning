from sklearn import tree,svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
#[height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],[177, 70, 40], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'male']
prediction_sample = [[159, 55, 37], [171, 75, 42],[193,68,45]]
prediction_output = ['female','male','male']
#Decision Tress Classifier
classifier = tree.DecisionTreeClassifier()

classifier = classifier.fit(X,Y)
output = classifier.predict(prediction_sample)

print(output)

#Support vector Classifier
clf1 = svm.SVC()

clf1 = clf1.fit(X,Y)
out1 = clf1.predict(prediction_sample)
print(out1)

#Guassian classifier Naive Bayes
gnb =  GaussianNB()

gnb = gnb.fit(X,Y)
out2 = gnb.predict(prediction_sample) 
print(out2)

#Gredient Decent classifier
clf3 = SGDClassifier(loss='hinge',penalty='l2',max_iter=5)
clf3.fit(X,Y)
out3 = clf3.predict(prediction_sample)
print(out3)

#Random Forest
