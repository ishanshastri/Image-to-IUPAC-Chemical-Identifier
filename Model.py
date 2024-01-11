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
from scipy import spatial
import cv2
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import array_to_img

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

def _get_closest_embedding(vector, embeds):
    #euclidean = lambda x, y: 
    #distances = {embed:spatial.distance.cosine(vector, embed) for embed in embeds}
    #sorted_dists = sorted(distances.items(), key=lambda x:x[1])
    distances = [(embed, spatial.distance.cosine(vector, embed)) for embed in embeds]
    sorted_dists = sorted(distances, key=lambda x:x[1])
    return sorted_dists[0]

def generator(inputs, outputs, batchSize):
    """
    Generating function for input images (random chem diagrams from pubchem) and corresponding graph embedding outputs; generates batches
    """
    N = len(inputs)
    ind = 0
    while True:
        #yield inputs[ind:(ind+batchSize)], [_getEmbedding(i, outputs) for i in range(ind, (ind+batchSize))]
        yield [tf.keras.preprocessing.image.load_img(inputs[i], True).resize((150, 150)) for i in range(ind, (ind+batchSize))],[_getEmbedding(i, outputs) for i in range(ind, (ind+batchSize))]
        ind += batchSize
        #ind += batchSize
        if ind + batchSize > N:
            ind = 0

#Retrieve data
Data = GetData(DATA_PATH)

#Split training and test data
TrainX = np.asarray([img_to_array(tf.keras.preprocessing.image.load_img(Data['Images'][i], True).resize((150, 150))) for i in range(0, 1000)])
TrainY = np.array([_getEmbedding(i, Data.drop(['Images'], axis=1)) for i in range(0, 1000)])

TestX = np.asarray([img_to_array(tf.keras.preprocessing.image.load_img(Data['Images'][i], True).resize((150, 150))) for i in range(1000, 1249)])
TestY = np.asarray([_getEmbedding(i, Data.drop(['Images'], axis=1)) for i in range(1000, 1249)])

AllEmbeddings = np.concatenate((TrainY, TestY))

#-------------Image Loading
loaded_img = tf.keras.preprocessing.image.load_img('ImageData/uh1.png', True)#colour_mode="grayscale")
#v_loadimg = np.vectorize(tf.keras.preprocessing.image.load_img)
#v_loadimg(['ImageData/uh0.png', 'ImageData/uh1.png'], True)

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

'''training model USING GENERATOR 
history = model.fit(generator(Data['Images'], Data.drop(['Images'], axis=1), 32),
 validation_data = (TestX, TestY), steps_per_epoch = 1000 // 32,
 epochs = 10)


#Train model
history = model.fit(TrainX, TrainY,
 validation_data = (TestX, TestY), 
 epochs = 10)
'''
history = model.fit(TrainX, TrainY,
 validation_data = (TestX, TestY), 
 epochs = 1)

# Running samples
#sample_chemical_embed = AllEmbeddings[50]
sample_chemical_diagram = TrainX[50]
#model_prediction = model.predict([sample_chemical_diagram])
model_prediction = model(np.reshape(sample_chemical_diagram, (1, 150, 150)))
closest_embed = _get_closest_embedding(vector=model_prediction[0], embeds=AllEmbeddings)[0]

# TODO retrieve corresponding iupac, compare w/ actual iupac
 
#def loadAndTrainLSTM()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
