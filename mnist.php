<?php

require_once 'src/Dataset/Dataset.php';
require_once 'src/Dataset/DatasetReader.php';
require_once 'src/NeuralNetwork.php';

use MNIST\Dataset\Dataset;
use MNIST\Dataset\DatasetReader;
use MNIST\NeuralNetwork;

$BATCH_SIZE = 10;
$STEPS = 1000;

// Load Training Dataset
$trainImagePath = 'data/train-images-idx3-ubyte';
$trainLabelPath = 'data/train-labels-idx1-ubyte';

$trainDataset = DatasetReader::fromFiles($trainImagePath, $trainLabelPath);

// Load Test Dataset
$testImagePath = 'data/t10k-images-idx3-ubyte';
$testLabelPath = 'data/t10k-labels-idx1-ubyte';

$testDataset = DatasetReader::fromFiles($testImagePath, $testLabelPath);

// Accuracy Evaluation
function calculate_accuracy(NeuralNetwork $neuralNetwork, Dataset $dataset)
{
    $size = $dataset->getSize();

    for ($i = 0, $correct = 0; $i < $size; $i++) {
        $image = $dataset->getImage($i);
        $label = $dataset->getLabel($i);

        $activations = $neuralNetwork->hypothesis($image);

        // Our prediction has the maximum probability
        $prediction = array_search(max($activations), $activations);

        if ($prediction == $label) {
            $correct++;
        }
    }

    return $correct / $size;
}

// Create Network
$neuralNetwork = new NeuralNetwork();

// Begin Training
$batches = $trainDataset->getSize() / $BATCH_SIZE;

for ($i = 0; $i < $STEPS; $i++) {
    $batch = $trainDataset->getBatch($BATCH_SIZE, $i % $batches);

    $loss = $neuralNetwork->trainingStep($batch, 0.5);
    $averageLoss = $loss / $batch->getSize();

    $accuracy = calculate_accuracy($neuralNetwork, $testDataset);

    echo 'Average Loss: ' . $averageLoss . PHP_EOL;
    echo 'Accuracy: ' . $accuracy . PHP_EOL . PHP_EOL;
}
