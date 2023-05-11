#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[89]:


#csv파일을 읽어와서 npy로 변환 / 데이터셋 생성과정에서 npy로 변환하여 저장하는 것이 좋을듯
e_data = pd.read_csv('./dataset/e_data.csv', header=None)
i_data = pd.read_csv('./dataset/i_data.csv', header=None)
l_data = pd.read_csv('./dataset/l_data.csv', header=None)
o_data = pd.read_csv('./dataset/o_data.csv', header=None)
u_data = pd.read_csv('./dataset/u_data.csv', header=None)
v_data = pd.read_csv('./dataset/v_data.csv', header=None)
y_data = pd.read_csv('./dataset/y_data.csv', header=None)

e_train_data = np.array(e_data)
i_train_data = np.array(i_data)
l_train_data = np.array(l_data)
o_train_data = np.array(o_data)
u_train_data = np.array(u_data)
v_train_data = np.array(v_data)
y_train_data = np.array(y_data)

#csv파일을 npy로 변환하여 concatnate

data = np.concatenate((e_train_data,
                       i_train_data,
                       l_train_data,
                       o_train_data,
                       u_train_data,
                       v_train_data,
                       y_train_data), axis=0)

gesture = ['e', 'i', 'l', 'o', 'u', 'v', 'y']

print(data.shape)
print(len(gesture))


# In[90]:


x_data = data[:,:-1]
labels = data[:,-1]

print(x_data.shape)
print(labels.shape)


# In[95]:


#one-hot encoding
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
y_data = enc.fit_transform(labels.reshape(-1,1)).toarray()

print(y_data.shape)

print(y_data[0])


# In[96]:


from sklearn.model_selection import train_test_split

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)

print(x_data_train.shape, x_data_test.shape)
print(y_data_train.shape, y_data_test.shape) 
print(x_data_train[0])
print(y_data_train[0])


# In[100]:


#KNN 모델 TEST
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(x_data_train, y_data_train)

y_pred = knn.predict(x_data_test)

y_pred1 = knn.predict([[20,36,22,13,2,3,7,4,3,20,150,9,33,138,9]]) #u

y_pred2 = knn.predict([[25,40,11,10,2,1,8,2,5,21,150,12,5,160,7]]) #v

print(knn.score(x_data_test, y_data_test))
print(y_pred1)
print(y_pred2) 


# In[138]:


#Sequential Model TEST
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt


model = Sequential()
model.add(Dense(64, input_shape=(15,), activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(7, activation='softmax'))

model.compile( loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


# In[ ]:


# history = model.fit(x_data_train, y_data_train, epochs=200, batch_size=8)

# test_loss, test_acc = model.evaluate(x_data_test, y_data_test)

# loss = history.history['loss']
# acc = history.history['accuracy']
# epochs = range(1, len(loss)+1)

# plt.plot(epochs, loss, 'r', label='Loss')
# plt.plot(epochs, acc, 'g', label='Accuracy')
# plt.legend()
# plt.xlabel('epochs')
# plt.ylabel('loss/acc')
# plt.show()

# print("loss : ", test_loss)
# print("acc : ", test_acc)

# y_pred1 = model.predict([[20,36,22,13,2,3,7,4,3,20,150,9,33,138,9]]) #u

# y_pred2 = model.predict([[25,40,11,10,2,1,8,2,5,21,150,12,5,160,7]]) #v

# print(np.round(y_pred1)) #[[0. 0. 0. 0. 1. 0. 0.]]
# print(np.round(y_pred2)) #[[0. 0. 0. 0. 0. 1. 0.]]


# In[139]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

modelpath = "./model/{epoch:02d}model.hdf5"

checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

history = model.fit(x_data_train, y_data_train, validation_split=0.25, epochs=10000, batch_size=8,
                    verbose=1, callbacks=[early_stopping_callback, checkpointer])

test_loss, test_acc = model.evaluate(x_data_test, y_data_test)

loss = history.history['loss']
acc = history.history['accuracy']
epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'r', label='Loss')
plt.plot(epochs, acc, 'g', label='Accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss/acc')
plt.show()

print("loss : ", test_loss)
print("acc : ", test_acc)

y_pred1 = model.predict([[20,36,22,13,2,3,7,4,3,20,150,9,33,138,9]]) #u

y_pred2 = model.predict([[25,40,11,10,2,1,8,2,5,21,150,12,5,160,7]]) #v

print(np.round(y_pred1)) #[[0. 0. 0. 0. 1. 0. 0.]]
print(np.round(y_pred2)) #[[0. 0. 0. 0. 0. 1. 0.]]


# In[137]:


del model

