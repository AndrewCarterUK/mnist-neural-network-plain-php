# MNIST Neural Network in PHP


This source code seeks to replicate the (now removed) [MNIST For ML Beginners](https://web.archive.org/web/20180801165522/https://www.tensorflow.org/versions/r1.1/get_started/mnist/beginners) tutorial from the Tensorflow website using straight forward PHP code. Hopefully, this example will make that tutorial a bit more manageable for PHP developers.

The task is to recognise digits, such as the ones below, as accurately as possible.

![MNIST digits](https://www.tensorflow.org/versions/r1.1/images/MNIST.png)

By [AndrewCarterUK ![(Twitter)](http://i.imgur.com/wWzX9uB.png)](https://twitter.com/AndrewCarterUK)

## Contents

- [mnist.php](mnist.php): Glue code that runs the algorithm steps and reports algorithm accuracy
- [Dataset.php](src/Dataset.php): Dataset container object
- [DatasetReader.php](src/DatasetReader.php): Retrieves images and labels from the MNIST dataset
- [NeuralNetwork.php](src/NeuralNetwork.php): Implements training and prediction routines for a simple neural network

## Usage

```sh
php mnist.php
```

## Description

The neural network implemented has one output layer and no hidden layers. Softmax activation is used, and this ensures that the output activations form a probability vector corresponding to each label. The cross entropy is used as a loss function.

This algorithm can achieve an accuracy of around 92% (with a batch size of 100 and 1000 training steps). That said, you are likely to get bored well before that point with PHP.

## Expected Output

```
Loading training dataset... (may take a while)
Loading test dataset... (may take a while)
Starting training...
Step 0001	Average Loss 4.12	Accuracy: 0.19
Step 0002	Average Loss 3.21	Accuracy: 0.23
Step 0003	Average Loss 2.59	Accuracy: 0.32
Step 0004	Average Loss 2.43	Accuracy: 0.36
Step 0005	Average Loss 1.87	Accuracy: 0.45
Step 0006	Average Loss 2.06	Accuracy: 0.47
Step 0007	Average Loss 1.67	Accuracy: 0.51
Step 0008	Average Loss 1.81	Accuracy: 0.46
Step 0009	Average Loss 1.74	Accuracy: 0.55
Step 0010	Average Loss 1.24	Accuracy: 0.56
...
```

![training evolution](https://res.cloudinary.com/andrewcarteruk/image/upload/v1523189356/training-evolution_hhbsfb.png)
