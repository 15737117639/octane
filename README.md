
<h1>Octane number prediction using neural network</h1>
2D fingerprint-based artificial QSAR using neural networks approach

X axis - experimental octane numbers, Y axis - our model:
r<sup>2</sup> = 0.9267
<img src=https://raw.githubusercontent.com/Blyschak/octane/master/data/graphs/experiment-vs-model.png></img>

As you can see there is correlation between our model and experimental data set.
<h1>Model</h1>
The NN impementation module provides a class Model which you can use to easily create your NN-based model

Prototype: 
<p>
<code>
Model(shape, learning_rate=0.1, epochs=10)
</code>

Also OpenBabel Python API was used to parse SMILES structures

NN used in this project: shape = [1024, 800, 1], learning rate = 0.0001, epochs = 1000, a small(for now) training data set with 27 molecules
