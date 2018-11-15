import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegressionCV as lr
from sklearn.decomposition import PCA as pca
from sklearn.preprocessing import StandardScaler as sc

# data preparation
# read original data from file
oriData = pd.read_csv("../20yrs_data.csv", na_values = np.nan)

# check if null exist
nullValue = oriData[oriData.isnull().values==True]
nullIndex = nullValue.index.tolist()
# drop null rows
cleanData = oriData.drop(nullIndex)
# preprocess time
tempTime = cleanData['Time'].str.split(':', expand=True).pop(0)
tempTime = pd.DataFrame(tempTime,dtype=np.float)
cleanData = cleanData.drop(columns = ['Time'])
cleanData.insert(3,'Time',tempTime)

# change type to numpy array
data = cleanData.values
c = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18]

# get features and labels
label = data[:, 12]
data = data[:, c]

# scale data
data = sc.fit_transform(data)
pca = pca(n_components=12)
fit = pca.fit_transform(data)
data = pd.DataFrame(data = fit)
print(data)

# set up the model
logisticRegressionInstance = lr()
logisticRegressionInstance.fit(data, label.astype('int'))

print(label[182100:182114])
print(logisticRegressionInstance.get_params())
print(logisticRegressionInstance.predict(data[182100:182114]))
print(logisticRegressionInstance.predict_proba(data[182100:182114])[:, 1])

