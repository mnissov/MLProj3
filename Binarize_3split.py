import pandas as pd
import numpy as np

from initData import *
np.set_printoptions(threshold=np.nan)

newAttrNames = np.array(["wage1", "wage2","wage3","hours1","hours2",\
                              "hours3","iq1","iq2","iq3","educ1","educ2",\
                              "educ3","exper1","exper2","exper3","tenure1",\
                              "tenure2","tenure3","age1","age2","age3",\
                              "not_black", "black"])

matrix = np.mat(dfMatrix[1:])

y_len = len(matrix[0:,0])
attr_len = len(newAttrNames)

# Create new empty matrix
newMatrix = np.zeros(shape=(y_len,attr_len))

# Wages - binerize by percentile
wages_33_percentile = np.percentile(matrix[0:,0], (1/3)*100)
wages_67_percentile = np.percentile(matrix[0:,0], (2/3)*100)

for i in range(y_len):
    if matrix[i,0] < wages_33_percentile:
        newMatrix[i,0] = 1
    elif wages_33_percentile <= matrix[i,0] < wages_67_percentile:
        newMatrix[i,1] = 1
    elif wages_67_percentile <= matrix[i,0]:
        newMatrix[i,2] = 1
    else:
        print("Something whent wrong with wages")
        
# hours - binerize by percentile
hours_33_percentile = np.percentile(matrix[0:,1], (1/3)*100)
hours_67_percentile = np.percentile(matrix[0:,1], (2/3)*100)

for i in range(y_len):
    if matrix[i,1] < hours_33_percentile:
        newMatrix[i,3] = 1
    elif hours_33_percentile <= matrix[i,1] < hours_67_percentile:
        newMatrix[i,4] = 1
    elif hours_67_percentile <= matrix[i,1]:
        newMatrix[i,5] = 1
    else:
        print("Something whent wrong with hours")
        
# iq - binerize by percentile
iq_33_percentile = np.percentile(matrix[0:,2], (1/3)*100)
iq_67_percentile = np.percentile(matrix[0:,2], (2/3)*100)

for i in range(y_len):
    if matrix[i,2] < iq_33_percentile:
        newMatrix[i,6] = 1
    elif iq_33_percentile <= matrix[i,2] < iq_67_percentile:
        newMatrix[i,7] = 1
    elif iq_67_percentile <= matrix[i,2]:
        newMatrix[i,8] = 1
    else:
        print("Something whent wrong with iq")

# educ - binerize by percentile
educ_33_percentile = np.percentile(matrix[0:,3], (1/3)*100)
educ_67_percentile = np.percentile(matrix[0:,3], (2/3)*100)

for i in range(y_len):
    if matrix[i,3] < educ_33_percentile:
        newMatrix[i,9] = 1
    elif educ_33_percentile <= matrix[i,3] < educ_67_percentile:
        newMatrix[i,10] = 1
    elif educ_67_percentile <= matrix[i,3]:
        newMatrix[i,11] = 1
    else:
        print("Something whent wrong with educ")

# exper - binerize by percentile
exper_33_percentile = np.percentile(matrix[0:,4], (1/3)*100)
exper_67_percentile = np.percentile(matrix[0:,4], (2/3)*100)

for i in range(y_len):
    if matrix[i,4] < exper_33_percentile:
        newMatrix[i,12] = 1
    elif exper_33_percentile <= matrix[i,4] < exper_67_percentile:
        newMatrix[i,13] = 1
    elif exper_67_percentile <= matrix[i,4]:
        newMatrix[i,14] = 1
    else:
        print("Something whent wrong with exper")

# tenure - binerize by percentile
tenure_33_percentile = np.percentile(matrix[0:,5], (1/3)*100)
tenure_67_percentile = np.percentile(matrix[0:,5], (2/3)*100)

for i in range(y_len):
    if matrix[i,5] < tenure_33_percentile:
        newMatrix[i,15] = 1
    elif tenure_33_percentile <= matrix[i,5] < tenure_67_percentile:
        newMatrix[i,16] = 1
    elif tenure_67_percentile <= matrix[i,5]:
        newMatrix[i,17] = 1
    else:
        print("Something whent wrong with tenure")
        
# age - binerize by percentile
age_33_percentile = np.percentile(matrix[0:,6], (1/3)*100)
age_67_percentile = np.percentile(matrix[0:,6], (2/3)*100)

for i in range(y_len):
    if matrix[i,6] < age_33_percentile:
        newMatrix[i,18] = 1
    elif age_33_percentile <= matrix[i,6] < age_67_percentile:
        newMatrix[i,19] = 1
    elif age_67_percentile <= matrix[i,6]:
        newMatrix[i,20] = 1
    else:
        print("Something whent wrong with age")
        
# black - binerize by percentile
for i in range(y_len):
    if matrix[i,7] < 1:
        newMatrix[i,21] = 1
    elif 1 == matrix[i,7]:
        newMatrix[i,22] = 1
    else:
        print("Something whent wrong with black")


dictionary = {}

for i in range(y_len):
    dictionary[i] = []
    for a in range(attr_len):
        if newMatrix[i,a] == 1:
            dictionary[i].append(a+1)
        else:
            pass

df = pd.DataFrame.from_dict(dictionary, orient="index")


np.savetxt(r'data.txt', df.values, fmt='%d', delimiter=",")
