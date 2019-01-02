#loading dependancies
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

#setting random seeds
np.random.seed(0)

iris_data = load_iris()
df = pd.DataFrame(iris_data.data, columns = iris_data.feature_names)
print(df.head)

df['species'] = pd.Categorical.from_codes(iris_data.target,iris_data.target_names)
print(df.head())

df['is_train'] = np.random.uniform(0,1,len(df)) <= 0.75
print(df.head())

#Divide into training and testing
train = df[df['is_train'] == True]
test = df[df['is_train'] == False]
features = df.columns[:4]
print(features)
y = pd.factorize(train['species'])[0]
print(y)
clf = RandomForestClassifier(n_jobs = 2,random_state = 0)
clf.fit(train[features],y)
clf.predict(test[features])
clf.predict_proba(test[features])[0:10]
preds = iris_data.target_names[clf.predict(test[features])]
preds[0:5]
print(test['species'].head())
print(pd.crosstab(test['species'],preds,rownames = ['Actual_Species'],colnames = ['Predicted_Species']))