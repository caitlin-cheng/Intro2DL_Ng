# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:46:35 2018

@author: Xiaoyin
"""

import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


np.random.seed(1);
X, Y = load_planar_dataset()
print(X.shape)
print(Y.shape)
plt.scatter(X[0,:], X[1,:], c=Y[0, :], s=40, cmap=plt.cm.Spectral)

m = X.shape[1]
print(m)

clf = sklearn.linear_model.LogisticRegressionCV(); 
clf.fit(X.T, Y.T);

print(clf.predict)

plot_decision_boundary(lambda x : clf.predict(x), X, Y)
plt.title("Logistic Regression")