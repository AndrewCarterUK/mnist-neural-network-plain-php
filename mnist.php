<?php

require_once 'src/Dataset.php';
require_once 'src/DatasetReader.php';
require_once 'src/NeuralNetwork.php';

use MNIST\Dataset;
use MNIST\DatasetReader;
use MNIST\NeuralNetwork;

$BATCH_SIZE = 100;
$STEPS = 1000;

// Load Training Dataset
$trainImagePath = 'data/train-images-idx3-ubyte';
$trainLabelPath = 'data/train-labels-idx1-ubyte';

echo 'Loading training dataset... (may take a while)' . PHP_EOL;
$trainDataset = DatasetReader::fromFiles($trainImagePath, $trainLabelPath);

// Load Test Dataset
$testImagePath = 'data/t10k-images-idx3-ubyte';
$testLabelPath = 'data/t10k-labels-idx1-ubyte';

echo 'Loading test dataset... (may take a while)' . PHP_EOL;
$testDataset = DatasetReader::fromFiles($testImagePath, $testLabelPath);

// Accuracy Evaluation
function calculate_accuracy(NeuralNetwork $neuralNetwork, Dataset $dataset)
{
    $size = $dataset->getSize();

    // Loop through all the training examples
    for ($i = 0, $correct = 0; $i < $size; $i++) {
        $image = $dataset->getImage($i);
        $label = $dataset->getLabel($i);

        $activations = $neuralNetwork->hypothesis($image);

        // Our prediction is index containing the maximum probability
        $prediction = array_search(max($activations), $activations);

        if ($prediction == $label) {
            $correct++;
        }
    }

    // Percentage of correct predictions is the accuracy
    return $correct / $size;
}

// Create Network
$neuralNetwork = new NeuralNetwork();

// Begin Training
$batches = $trainDataset->getSize() / $BATCH_SIZE;

echo 'Starting training...' . PHP_EOL;

for ($i = 0; $i < $STEPS; $i++) {
    $batch = $trainDataset->getBatch($BATCH_SIZE, $i % $batches);

    $loss = $neuralNetwork->trainingStep($batch, 0.5);
    $averageLoss = $loss / $batch->getSize();

    $accuracy = calculate_accuracy($neuralNetwork, $testDataset);

    printf("Step %04d\tAverage Loss %.2f\tAccuracy: %.2f\n", $i + 1, $averageLoss, $accuracy);
}
