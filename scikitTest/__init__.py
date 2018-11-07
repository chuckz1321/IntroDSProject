import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as lr

#read the file
filePath = "../20yrs_data.csv"
pwd = os.getcwd()
os.chdir(os.path.dirname(filePath))
trainData = pd.read_csv(os.path.basename(filePath))
os.chdir(pwd)
data = trainData.values
#set up the model
print(data[139830:139850,12])
c = [4,5,6,7,8,9,10]
logisticRegressionInstance = lr()
logisticRegressionInstance.fit(data[1:1000,c],data[1:1000,12].astype('int'))

print(logisticRegressionInstance.predict_proba(data[139830:139850,c])[:,1])

