# Furkan Hatipoglu
#090130347


#task1
import os
    
assert [os.path.isdir("../"+x) for x in ['homeworks','data','images','other']] == [True, True, True, True]
assert os.path.isfile("../homeworks/HW1.ipynb")


#task2
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import contingency_matrix
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

#task3
a = np.matrix([[0,1],[1,0]])
assert np.array_equal(a*a, np.eye(2)) == True

#task4
a = np.array(stats.norm.rvs(size=3,random_state=1234))
assert np.abs((a - np.array([0.47143516, -1.19097569,  1.43270697]))).sum() < 1e-7

#task5
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                   sep=',',
                   header=None)

iris.to_csv("../data/iris.csv", sep=',', header=False)

assert iris.shape == (150, 5)
assert os.path.isfile('../data/iris.csv') == True

#task6
X_train,X_test, y_train, y_test = train_test_split(iris.iloc[:,0:4],iris.iloc[:,4],test_size=0.33)
model = KMeans(n_clusters=3, random_state=42).fit(X_train)
result = contingency_matrix(y_test,model.predict(X_test))

assert X_train.shape == (100,4)
assert model.random_state == 42
assert result.shape == (3,3)

#task7
iris = load_iris()

x_index = 3
y_index = 2

formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])

plt.tight_layout()
plt.savefig('../images/HW1-task6.png')
plt.show()

assert os.path.isfile('../images/HW1-task6.png')