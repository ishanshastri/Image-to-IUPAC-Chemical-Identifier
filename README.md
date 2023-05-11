# Image-to-IUPAC-Chemical-Identifier
A CNN model to identify chemicals from their diagrams (produce IUPAC names from images). Built using the graph representations of molecules

Machine Learning Tools: Tensorflow 

Graph Tools: Networkx (Graph toolkit), Graph2Vec (Graph embedding toolkit)

# Architecture Overview
The model takes a PNG image as input (resized to 150 x 150 pixels), and produces a graph embedding as output. This is then used to obtain a graph representation of a molecule, from which the IUPAC name is constructed and otuputted.

**Current neural network architecture (tensorflow model summary):**

![image](https://github.com/ishanshastri/Image-to-IUPAC-Chemical-Identifier/assets/94653377/5d4e2a7f-039b-4a08-87e0-ad4557e595e5)

