# Neural Networks :P
Recalling our logistic regression, we can imagine perceptron as a logistic regression, because perceptron is a single neuron and it works like this:

A single neuron has:
- Dendirtes: that are our input pipes
- Axon: our output
- Cell body: Computation of sigmoid function

As we can see, if we have a single neuron, we are computating a logistic regression

#
A neural network has many "logistic regression units", so, we have a parameters theta matrix.

Introducing a little of notation, sigmoid function is named "activation function", theta parameters are named "weights".

And as we told previously, we have several units distributed by fields.

There are 3 types of fields:
1. Input field
2. Hidden field
3. Output field

- To specify each field, we use superscript ***i***
- To specify each unit by field, we user subscript ***j***
- First and last field has not superscript

Weight (parameters) matrices controlling function mapping from layer j to layer j+1: Each field has their own theta matrix

To compute dimensions of theta matrix we use:
- s(j+1) * (sj + 1)

#

For this excercise we are going to use a training set, which has 20x20-pixels images of handwritten numbers, represented as vectors of 400 positions. Using our neural network and oneVsAll algorithm we can train and then predict given a new example which number is it.

![oneVsAll](https://github.com/JhonSanz/oneVsAll/blob/master/number_2.png?raw=true) It looks like number: 2