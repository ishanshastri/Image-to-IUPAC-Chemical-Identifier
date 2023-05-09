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
        yield inputs[ind:(ind+batchSize)], [_getEmbedding(i, outputs) for i in range(ind, (ind+batchSize))]
        ind += batchSize
        if ind + batchSize > N:
            ind = 0

#TEST:
#For single datapoint
Data = GetData(DATA_PATH)
embedding = _getEmbedding(0, Data.drop(['Images'], axis=1))
input = Data['Images'][0]
print(len(embedding), input, embedding)

#for multiple datapoints (5)
embeds = Data.drop(['Images'], axis=1)
print([(tf.keras.preprocessing.image.load_img(Data['Images'][i], True), _getEmbedding(i, embeds)) for i in range(0, 5)])

#-------------Image Loading
loaded_img = tf.keras.preprocessing.image.load_img('uh.png', True)#colour_mode="grayscale")
print(loaded_img)