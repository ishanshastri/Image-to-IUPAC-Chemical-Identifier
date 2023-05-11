# Image-to-IUPAC-Chemical-Identifier
A CNN model to identify chemicals from their diagrams (produce IUPAC names from images). Built using the graph representations of molecules

Machine Learning Tools: Tensorflow 

Graph Tools: Networkx (Graph toolkit), Graph2Vec (Graph embedding toolkit)

# Architecture Overview
The model takes a PNG image as input (resized to 150 x 150 pixels), and produces a graph embedding as output. This is then used to obtain a graph representation of a molecule, from which the IUPAC name is constructed and otuputted.

# Current neural network architecture (tensorflow model summary):
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 148, 148, 32)      320

 max_pooling2d (MaxPooling2D  (None, 74, 74, 32)       0
 )

 conv2d_1 (Conv2D)           (None, 72, 72, 64)        18496

 max_pooling2d_1 (MaxPooling  (None, 36, 36, 64)       0
 2D)

 conv2d_2 (Conv2D)           (None, 34, 34, 64)        36928

 conv2d_3 (Conv2D)           (None, 32, 32, 128)       73856

 flatten (Flatten)           (None, 131072)            0

 dense (Dense)               (None, 128)               16777344

 dense_1 (Dense)             (None, 128)               16512

=================================================================

Total params: 16,923,456

Trainable params: 16,923,456

Non-trainable params: 0
