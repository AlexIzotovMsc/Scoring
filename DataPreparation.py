import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:\PyScripts\PyData\Data2.txt")
print(df.head(5))

y = df.iloc[:, 13].values
X1 =df.iloc[:, 1].values
X2 =df.iloc[:, 2].values
X3 =df.iloc[:, 3].values
X4 =df.iloc[:, 4].values
X5 =df.iloc[:, 5].values
X6 =df.iloc[:, 6].values
X7 =df.iloc[:, 7].values
X8 =df.iloc[:, 8].values
X9 =df.iloc[:, 9].values
X10 =df.iloc[:, 10].values
X11 =df.iloc[:, 11].values
X12 =df.iloc[:, 12].values

# Encoding binary features and get the dummies
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import np_utils
#X4
encoder = LabelEncoder()
encoder.fit(X4)
encoded_X4 = encoder.transform(X4)
dummy_X4 = np_utils.to_categorical(encoded_X4) 

#X5
X5 = X5.reshape(-1, 1)
encoder.fit(X5)
encoded_X5 = encoder.transform(X5)
dummy_X5 = np_utils.to_categorical(encoded_X5)

#X6
X6 = X6.reshape(-1, 1)
encoder.fit(X6)
encoded_X6 = encoder.transform(X6)
dummy_X6 = np_utils.to_categorical(encoded_X6)

#X7
X7 = X7.reshape(-1, 1)
encoder.fit(X7)
encoded_X7 = encoder.transform(X7)
dummy_X7 = np_utils.to_categorical(encoded_X7)

#X8
X8 = X8.reshape(-1, 1)
encoder.fit(X8)
encoded_X8 = encoder.transform(X8)
dummy_X8 = np_utils.to_categorical(encoded_X8)

#X10
X10 = X10.reshape(-1, 1)
encoder.fit(X10)
encoded_X10 = encoder.transform(X10)
dummy_X10 = np_utils.to_categorical(encoded_X10)

#X11
X11 = X11.reshape(-1, 1)
encoder.fit(X11)
encoded_X11 = encoder.transform(X11)
dummy_X11 = np_utils.to_categorical(encoded_X11)

#X12
X12 = X12.reshape(-1, 1)
encoder.fit(X12)
encoded_X12 = encoder.transform(X12)
dummy_X12 = np_utils.to_categorical(encoded_X12)


#######
XI = np.vstack((X1, X2, X3, X9, encoded_X4, encoded_X7, encoded_X12)).T.astype(np.float64)
Xdummies = np.concatenate(([dummy_X5, dummy_X6, dummy_X8, dummy_X10, dummy_X11]), axis=1)

X =  np.concatenate(([XI, Xdummies]), axis=1)

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
