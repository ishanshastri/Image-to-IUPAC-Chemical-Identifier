#TF etc (and friends)
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import random
import networkx as nx
from karateclub import Graph2Vec
import scipy
import cv2

#Enable eager execution TODO
'''
import tensorflow as tf
tf.enable_eager_execution()
tfe = tf.contrib.eager
'''
DATA_PATH = 'CompiledDataset/relevant_data.csv'

#Retrieve dataframe from stored csv file
def GetData(file):
    """
    Retrieve data from file, return pandas dataframe
    """
    d = pd.read_csv(file)
    Data = d['Images']
    Data = pd.concat([Data, d[[str(i) for i in range(128)]]], axis="columns")
    return Data

def _getEmbedding(index, df):
    """
    Return an embedding from dataframe (consisting of only the embeddings)
    """
    return df.iloc[index].to_numpy()

def generator(inputs, outputs, batchSize):
    """
    Generating function for input images (random chem diagrams from pubchem) and corresponding graph embedding outputs; generates batches
    """
    N = len(inputs)
    ind = 0
    while True:
        #yield inputs[ind:(ind+batchSize)], [_getEmbedding(i, outputs) for i in range(ind, (ind+batchSize))]
        yield [tf.keras.preprocessing.image.load_img(inputs[i], True).resize(150, 150) for i in range(ind, (ind+batchSize))],[_getEmbedding(i, outputs) for i in range(ind, (ind+batchSize))]
        ind += batchSize
        #ind += batchSize
        if ind + batchSize > N:
            ind = 0

#TEST:
#For single datapoint
Data = GetData(DATA_PATH)

#Split training and test data
TestX = [tf.keras.preprocessing.image.load_img(Data['Images'][i], True).resize((150, 150)) for i in range(1000, 1249)]
TestY = [_getEmbedding(i, Data.drop(['Images'], axis=1)) for i in range(1000, 1249)]

#print(TestX)
#print(TestY)

#-------------Image Loading
loaded_img = tf.keras.preprocessing.image.load_img('ImageData/uh1.png', True)#colour_mode="grayscale")
#v_loadimg = np.vectorize(tf.keras.preprocessing.image.load_img)
#v_loadimg(['ImageData/uh0.png', 'ImageData/uh1.png'], True)
'''
print(loaded_img.resize((30, 30)).size)
loaded_img = loaded_img.resize((150, 150))
loaded_img.save('resized_uh.jpg')
'''
#-----------MODEL
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128))

model.summary()

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
'''
history = model.fit(train_images, train_labels, epochs=10)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

model.fit_generator(generator(Data['Images'], trainY, batch_size = 32),
 validation_data = (testX, testY), steps_per_epoch = len(trainX) // 32,
 epochs = 10)'''