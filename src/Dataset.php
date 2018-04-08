<?php

namespace MNIST;

class Dataset
{
    // All MNIST images are 28x28 and values 0-9
    const IMAGE_ROWS = 28;
    const IMAGE_COLS = 28;
    const IMAGE_SIZE = 28 * 28;
    const LABELS = 10;

    private $images;
    private $labels;
    private $size;

    public function __construct(array $images, array $labels)
    {
        if (count($images) != count($labels)) {
            throw new \Exception('Must have the same number of images and labels');
        }

        $this->images = $images;
        $this->labels = $labels;
        $this->size = count($images);
    }

    public function getImage($index): array
    {
        return $this->images[$index];
    }

    public function getLabel($index): int
    {
        return $this->labels[$index];
    }

    public function getSize(): int
    {
        return $this->size;
    }

    /**
     * Retrieve a subset of the dataset as a batch.
     */
    public function getBatch(int $size, int $number): Dataset
    {
        $offset = $size * $number;

        $images = array_slice($this->images, $offset, $size);
        $labels = array_slice($this->labels, $offset, $size);

        return new self($images, $labels);
    }
}
