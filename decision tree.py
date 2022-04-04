import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import MultiLabelBinarizer
import re

batch_size = 8
epochs = 40
num_classes = 3

# import data
df1 = pd.read_excel('coviddata2.xlsx', dtype="str", index_col=None)
df1.to_csv('coviddata2.csv', encoding='utf-8', index=False)

# convert type
df1.iloc[:, 0] = df1.iloc[:, 0].astype('float32') / 255
df1.iloc[:, 2] = df1.iloc[:, 2].astype('float32') / 255
df1.iloc[:, 5:24] = df1.iloc[:, 5:24].astype('float32') / 255

# normalization
xmax, xmin = df1.iloc[:, 0].max(), df1.iloc[:, 0].min()
df1.iloc[:, 0] = (df1.iloc[:, 0] - xmin) / (xmax - xmin)

xmax, xmin = df1.iloc[:, 2].max(), df1.iloc[:, 2].min()
df1.iloc[:, 2] = (df1.iloc[:, 2] - xmin) / (xmax - xmin)

xmax, xmin = df1.iloc[:, 5:24].max(), df1.iloc[:, 5:24].min()
df1.iloc[:, 5:24] = (df1.iloc[:, 5:24] - xmin) / (xmax - xmin)

# categorization
le = LabelEncoder()

df1.iloc[:, 3] = le.fit_transform(df1.iloc[:, 3])
df1.iloc[:, 1] = df1.iloc[:, 1].replace(to_replace='K', value=0)
df1.iloc[:, 1] = df1.iloc[:, 1].replace(to_replace='E', value=1)
df1.iloc[:, 4] = df1.iloc[:, 4].replace(to_replace='3', value=2)
df1.iloc[:, 4] = df1.iloc[:, 4].replace(to_replace='1', value=0)
df1.iloc[:, 4] = df1.iloc[:, 4].replace(to_replace='2', value=1)
df1.iloc[:, 24] = df1.iloc[:, 24].replace(to_replace='3', value=2)
df1.iloc[:, 24] = df1.iloc[:, 24].replace(to_replace='1', value=0)
df1.iloc[:, 24] = df1.iloc[:, 24].replace(to_replace='2', value=1)
df1.iloc[:, 25] = df1.iloc[:, 25].replace(to_replace='6', value=5)
df1.iloc[:, 25] = df1.iloc[:, 25].replace(to_replace='5', value=4)
df1.iloc[:, 25] = df1.iloc[:, 25].replace(to_replace='4', value=3)
df1.iloc[:, 25] = df1.iloc[:, 25].replace(to_replace='3', value=2)
df1.iloc[:, 25] = df1.iloc[:, 25].replace(to_replace='2', value=1)
df1.iloc[:, 25] = df1.iloc[:, 25].replace(to_replace='1', value=0)
df1.iloc[:, -1] = df1.iloc[:, -1].replace(to_replace='1', value=0)
df1.iloc[:, -1] = df1.iloc[:, -1].replace(to_replace='2', value=1)
df1.iloc[:, -1] = df1.iloc[:, -1].replace(to_replace='3', value=2)

# Ek hastalıklar - take as a list
sayı = []
for x in df1["EK HASTALIK"]:
    derle = re.compile("\w+", re.I)
    sayılar = derle.findall(x)
    sayı.append(sayılar)

# Ek hastalıklar 0-1 one hot encoding
df1.iloc[:, 27] = sayı
one_hot = MultiLabelBinarizer()
a = one_hot.fit_transform(df1.iloc[:, 27])

# Ek Hastalıklar - adding to the last columns of the data
columns = len(a[0])
i = 0
while i < columns:
    df1[i] = a[:, i]
    i = i + 1

# erasing data
del df1['EK HASTALIK']

# splitting data
y = np.array(df1['SONUÇ'])
del df1['SONUÇ']
x = np.array(df1)

# Reshape
x = x.reshape(x.shape[0], x.shape[1], 1)
y = to_categorical(y, num_classes=num_classes)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

#input_shape = (x.shape[1], 1)

# Model Training and Making Predictions
model = DecisionTreeClassifier(criterion='gini',
                               splitter='best',
                               min_samples_leaf=5,
                               max_depth=None)

model = model.fit(x_train, y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print('DecisionTreeClassifier accuracy score: {}'.format(accuracy))
