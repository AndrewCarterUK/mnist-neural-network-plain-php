<?php

namespace MNIST;

class NeuralNetwork
{
    const LABELS = 10;

    private $b;
    private $W;

    /**
     * Initialise the bias vector and weights as random values between 0 and 1.
     */
    public function __construct()
    {
        $this->b = [];
        $this->W = [];

        for ($i = 0; $i < self::LABELS; $i++) {
            $this->b[$i] = random_int(1, 1000) / 1000;
            $this->W[$i] = [];

            for ($j = 0; $j < Dataset::IMAGE_SIZE; $j++) {
                $this->W[$i][$j] = random_int(1, 1000) / 1000;
            }
        }
    }

    /**
     * The softmax layer maps an array of activations to a probability vector.
     */
    private function softmax(array $activations): array
    {
        // Normalising with the max activation makes the computation more numerically stable
        $max = max($activations);

        $activations = array_map(function ($a) use ($max) {
            return exp($a - $max);
        }, $activations);

        $sum = array_sum($activations);

        return array_map(function ($a) use ($sum) {
            return $a / $sum;
        }, $activations);
    }

    /**
     * Forward propagate through the neural network to calculate the activation
     * vector for an image.
     */
    public function hypothesis(array $image): array
    {
        $activations = [];

        for ($i = 0; $i < self::LABELS; $i++) {
            $activations[$i] = $this->b[$i];

            for ($j = 0; $j < Dataset::IMAGE_SIZE; $j++) {
                $activations[$i] += $this->W[$i][$j] * $image[$j];
            }
        }

        return $this->softmax($activations);
    }

    /**
     * Calculate the gradient adjustments on a single training example (image)
     * from the dataset.
     * 
     * Returns the contribution to the loss value from this example.
     */
    private function gradientUpdate(array $image, array &$bGrad, array &$WGrad, int $label): float
    {
        $activations = $this->hypothesis($image);

        for ($i = 0; $i < self::LABELS; $i++) {
            $bGradPart = ($i === $label) ? $activations[$i] - 1 : $activations[$i];

            for ($j = 0; $j < Dataset::IMAGE_SIZE; $j++) {
                $WGrad[$i][$j] += $bGradPart * $image[$j];
            }

            $bGrad[$i] += $bGradPart;
        }

        // Cross entropy
        return 0 - log($activations[$label]);
    }

    /**
     * Perform one step of gradient descent on the neural network using the
     * provided dataset.
     * 
     * Returns the total loss for the network on the provided dataset.
     */
    public function trainingStep(Dataset $dataset, float $learningRate): float
    {
        // Zero init the gradients
        $bGrad = array_fill(0, self::LABELS, 0);
        $WGrad = array_fill(0, self::LABELS, array_fill(0, Dataset::IMAGE_SIZE, 0));

        $totalLoss = 0;
        $size = $dataset->getSize();

        // Calculate the gradients and loss
        for ($i = 0; $i < $size; $i++) {
            $totalLoss += $this->gradientUpdate($dataset->getImage($i), $bGrad, $WGrad, $dataset->getLabel($i));
        }

        // Adjust the weights and bias vector using the gradient and the learning rate
        for ($i = 0; $i < self::LABELS; $i++) {
            $this->b[$i] -= $learningRate * $bGrad[$i] / $size;

            for ($j = 0; $j < Dataset::IMAGE_SIZE; $j++) {
                $this->W[$i][$j] -= $learningRate * $WGrad[$i][$j] / $size;
            }
        }

        return $totalLoss;
    }
}
