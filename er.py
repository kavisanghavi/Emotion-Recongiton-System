
# coding: utf-8

# In[1]:

import os
import time
import csv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import pandas as pd
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
#from skimage import data, filter
import pandas as pd
data=pd.read_csv("fer2013.csv")
#get_ipython().magic('matplotlib inline')


# In[2]:

image_x=[] #image pixel values
data.head()


# In[3]:

data.tail()


# In[4]:

x_train=data[:28709]


# In[5]:

x_train=data['pixels'].apply(lambda x: np.fromstring(x, sep=' '))
print(x_train[0].shape)


# In[ ]:




# In[6]:

X=np.vstack(x_train.values)/255
X=X.astype(np.float32)


# In[7]:

print (x_train[28708])


# In[8]:

print (type(X))
print (X[28708])
print (X.shape)


# In[9]:

x = data['pixels']
y = data['emotion']


from sklearn.preprocessing import MultiLabelBinarizer
y=MultiLabelBinarizer().fit_transform(y.reshape(-1,1))
print(len(y))



# In[10]:



# In[11]:

print (y[4])
print(x.shape)


# In[12]:

print(len(X),len(y))
x,x_validate,y,y_validate = X[:28708],X[28708:],y[:28708],y[28708:]
x = x.reshape(-1,48,48,1)
x_validate = x_validate.reshape(-1,48,48,1)
print(x.shape,y.shape,x_validate.shape,y_validate.shape)


# In[13]:

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.utils import shuffle
from keras.optimizers import SGD
import numpy as np


# In[ ]:

from keras.callbacks import EarlyStopping,ModelCheckpoint
model = Sequential()
model.add(Convolution2D(32,3,3,activation='relu',input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Convolution2D(32,3,3,activation='relu',input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(BatchNormalization())
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7,activation='softmax'))
sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer='adagrad',metrics=['accuracy'])

history=model.fit(x,y, validation_data=(x_validate,y_validate),nb_epoch=150, batch_size=128)


# In[ ]:

model.summary()
loss, accuracy = model.evaluate(x_validate, y_validate)
print('  vcx ')
print('accuracy: ', accuracy)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model_weights.h5")
model.save("model.h5")

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# In[ ]:




# In[ ]:




# In[ ]:



