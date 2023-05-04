#TF etc (and friends)
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import random
import cv2


#PubChem access API, chem tools et al
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import RDKFingerprint
from rdkit.Chem import rdMolDescriptors

#Test:
#c = pcp.Compound.from_cid(6819)
#c_smile = (c.isomeric_smiles)
#Dataframe:
#pcp.download('PNG', 'uh.png', 6819, 'cid', image_size='small', overwrite=True)
#pcp.download('PNG', 'uh.png', 6819, 'cid', overwrite=True)
#pcp.Compound.from_cid(6819)

'''
c = pcp.Compound.from_cid(6819)
c_smile = (c.isomeric_smiles)

#Time Analysis; API vs RealTime Chem Image Extraction
start_time = time.time()
pcp.download('PNG', 'uh.png', 6819, 'cid', image_size='small', overwrite=True)
print("--- %s API_sm ---" % (time.time() - start_time))

tart_time = time.time()
pcp.download('PNG', 'uh2.png', 6819, 'cid', overwrite=True)
print("--- %s API_lg ---" % (time.time() - start_time))

start_time = time.time()
c = pcp.Compound.from_cid(6819)
#mol = Chem.MolFromSmiles("CC1([C@H](N2[C@@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C")
mol = Chem.MolFromSmiles(c_smile)
Draw.MolToFile(mol, 'uh3.png')
print("--- %s RealTime ---" % (time.time() - start_time))
'''
#---------
#'''
#Retrieve compound object random.randint(0, 999999))
cid = random.randint(0, 9999999)
c = pcp.Compound.from_cid(cid)

#Get SMILE representation
c_smile = (c.isomeric_smiles)

#Download Image from CID
pcp.download('PNG', 'uh.png', cid, 'cid', image_size='large', overwrite=True)

#Get RDK representation of molecule
rdk_mol = Chem.MolFromSmiles(c_smile)

#Get Fingerprint as numpy array
fingerprint_c = np.array(RDKFingerprint(rdk_mol))

print(c.iupac_name, "F_Print: ", fingerprint_c, len(fingerprint_c))
#'''

#-------------Image Loading
#'''
loaded_img = tf.keras.preprocessing.image.load_img('uh.png', True)#colour_mode="grayscale")
print(loaded_img)

#Display image size
print(cv2.imread('uh.png').shape)

#'''

#--------------Net
#'''

#'''







