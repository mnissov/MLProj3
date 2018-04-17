# -*- coding: utf-8 -*-

import numpy as np
import xlrd
import pandas as pd
import math

from matplotlib.pyplot import figure, legend, subplot, plot, hist, title, imshow, yticks, cm, xlabel, ylabel, show, grid, boxplot
from scipy.linalg import svd
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import preprocessing
# Load xls sheet with data
#dataset = xlrd.open_workbook('wage2.xls').sheet_by_index(0)
#data = pd.get_dummies(dataset)


df = pd.read_excel('modified.xls', header = None)
doc = xlrd.open_workbook('modified.xls').sheet_by_index(0)

attributeNames = doc.row_values(0, 1, 8)
n = len(df.index)
df.reset_index()
df.reindex(index=range(0,n))

df.dropna(inplace=True)
dfMatrix = df.as_matrix()

y = dfMatrix[1:,0]
yMatrix = np.mat(y)

X = np.mat(np.empty((n-1,7)))

for i, col_id in enumerate(range(1,8)):
    X[:,i] = np.matrix(doc.col_values(col_id, 1, n)).T

classX = np.asarray(X)
stdX = preprocessing.scale(classX)
#N = len(y)
#M = len(attributeNames)

N, M = X.shape

classNames = ['Poor', 'Lower', 'Middle', 'Upper']

attributeNames = [
    'hours',
    'iq',
    'educ',
    'exper',
    'tenure',
    'age',
    'black'
    ]
    
    
classY = np.asarray(np.mat(np.empty((N))).T).squeeze()
for i in range(0,N):
    if y[i] <= np.percentile(y,25):
        classY[i] = 0
    elif y[i] <= np.percentile(y,50):
        classY[i] = 1
    elif y[i] <= np.percentile(y,75):
        classY[i] = 2
    else: 
        classY[i] = 3
        
C = len(classNames)

#boxplot(preprocessing.scale(classX))