import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, LabelBinarizer
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical

from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re

batch_size = 15
epochs = 50
num_classes = 3

# import data
df = pd.read_excel('coviddata.xlsx', dtype="str", index_col=None)
df.to_csv('coviddata.csv', encoding='utf-8', index=False)

# missing data
df.isnull().sum().sort_values(ascending=False)

# erasing data
df.drop(8, inplace=True)
df.drop(54, inplace=True)
df.drop(161, inplace=True)

# imputation
df.iloc[:, 2].fillna(value=0, inplace=True)
df.iloc[:, 3].fillna(value=2, inplace=True)
df.iloc[:, 4].fillna(value=1, inplace=True)
df.iloc[:, 5:29].fillna(value=0, inplace=True)
df.iloc[:, 29].fillna(value=1, inplace=True)

# convert type
df.iloc[:, 0] = df.iloc[:, 0].astype('float32') / 255
df.iloc[:, 2] = df.iloc[:, 2].astype('float32') / 255
df.iloc[:, 5:16] = df.iloc[:, 5:16].astype('float32') / 255
df.iloc[:, 3] = df.iloc[:, 3].astype('int')
df.iloc[:, 16] = df.iloc[:, 16].astype('int')

# normalization
xmax, xmin = df.iloc[:, 0].max(), df.iloc[:, 0].min()
df.iloc[:, 0] = (df.iloc[:, 0] - xmin) / (xmax - xmin)

xmax, xmin = df.iloc[:, 2].max(), df.iloc[:, 2].min()
df.iloc[:, 2] = (df.iloc[:, 2] - xmin) / (xmax - xmin)

xmax, xmin = df.iloc[:, 5:16].max(), df.iloc[:, 5:16].min()
df.iloc[:, 5:16] = (df.iloc[:, 5:16] - xmin) / (xmax - xmin)

# categorization
le = LabelEncoder()
lb = LabelBinarizer()

df.iloc[:, 1] = le.fit_transform(df.iloc[:, 1])
df.iloc[:, 3] = le.fit_transform(df.iloc[:, 3])
df.iloc[:, 17] = le.fit_transform(df.iloc[:, 17])
df.iloc[:, 18] = le.fit_transform(df.iloc[:, 18])
df.iloc[:, 19] = le.fit_transform(df.iloc[:, 19])
df.iloc[:, 20] = le.fit_transform(df.iloc[:, 20])
df.iloc[:, 21] = le.fit_transform(df.iloc[:, 21])
df.iloc[:, 22] = le.fit_transform(df.iloc[:, 22])
df.iloc[:, 23] = le.fit_transform(df.iloc[:, 23])
df.iloc[:, 24] = le.fit_transform(df.iloc[:, 24])
df.iloc[:, 25] = le.fit_transform(df.iloc[:, 25])
df.iloc[:, 26] = le.fit_transform(df.iloc[:, 26])
df.iloc[:, 27] = le.fit_transform(df.iloc[:, 27])
df.iloc[:, 28] = le.fit_transform(df.iloc[:, 28])
#df.iloc[:, 4] = lb.fit_transform(df.iloc[:, 4])
#df.iloc[:, 16] = lb.fit_transform(df.iloc[:, 16])
#df.iloc[:, 29] = lb.fit_transform(df.iloc[:, 29])

df.iloc[:, 4] = df.iloc[:, 4].replace(to_replace='3', value=2)
df.iloc[:, 4] = df.iloc[:, 4].replace(to_replace='1', value=0)
df.iloc[:, 4] = df.iloc[:, 4].replace(to_replace='2', value=1)
df.iloc[:, -1] = df.iloc[:, -1].replace(to_replace='1', value=0)
df.iloc[:, -1] = df.iloc[:, -1].replace(to_replace='2', value=1)
df.iloc[:, -1] = df.iloc[:, -1].replace(to_replace='3', value=2)

# splitting data
y = np.array(df['sonuç'])
del df['sonuç']
x = np.array(df)

print(y)

# Reshape
x = x.reshape(x.shape[0], x.shape[1], 1)
y = to_categorical(y, num_classes=num_classes)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3)  # Train:144 Validation:49 Test:84

x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  test_size=0.25)

# Create the model
model = Sequential()
model.add(
    Conv1D(128, 3, padding='same', activation='relu', input_shape=(29, 1)))
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(Conv1D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling1D((2), padding='same'))
model.add(Conv1D(16, 3, padding='same', activation='relu'))
model.add(MaxPooling1D((1), padding='same'))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

# Model Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Early Stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
mc = ModelCheckpoint('covid-predict-model.h5',
                     monitor='val_accuracy',
                     mode='max',
                     verbose=1,
                     save_best_only=True)

# Model Training
history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_val, y_val),
                    shuffle=True,
                    callbacks=[es, mc])

# Model Prediction
#x_test = x_test[1:2]
Y_pred = model.predict(x_test, verbose=2)
y_pred = np.argmax(Y_pred, axis=1)

#confusion matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
print(cm)

# Save the model to disk
model.save('covid-predict-model.h5')

print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#Plot loss curve
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()