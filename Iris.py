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