<?php

namespace MNIST;

class DatasetReader
{
    const MAGIC_IMAGE = 0x00000803;
    const MAGIC_LABEL = 0x00000801;

    /**
     * Build dataset from image and label file paths.
     */
    public static function fromFiles(string $imagePath, string $labelPath): Dataset
    {
        $images = self::readImages($imagePath);
        $labels = self::readLabels($labelPath);

        return new Dataset($images, $labels);
    }

    /**
     * Read MNIST image file.
     * 
     * Format: http://yann.lecun.com/exdb/mnist/
     */
    private static function readImages(string $imagePath): array
    {
        $stream = fopen($imagePath, 'rb');

        if (FALSE === $stream) {
            throw new \Exception('Could not open file: ' . $imagePath);
        }

        $images = [];

        try {
            $header = fread($stream, 16);

            $fields = unpack('Nmagic/Nsize/Nrows/Ncols', $header);

            if ($fields['magic'] !== self::MAGIC_IMAGE) {
                throw new \Exception('Invalid magic number: ' . $imagePath);
            } else if ($fields['rows'] != Dataset::IMAGE_ROWS) {
                throw new \Exception('Invalid number of image rows: ' . $imagePath);
            } else if ($fields['cols'] != Dataset::IMAGE_COLS) {
                throw new \Exception('Invalid number of image cols: ' . $imagePath);
            }

            for ($i = 0; $i < $fields['size']; $i++) {
                $imageBytes = fread($stream, $fields['rows'] * $fields['cols']);

                // Convert to float between 0 and 1
                $images[] = array_map(function ($b) {
                    return $b / 255;
                }, array_values(unpack('C*', $imageBytes)));
            }
        } finally {
            fclose($stream);
        }

        return $images;
    }

    /**
     * Read MNIST label file.
     * 
     * Format: http://yann.lecun.com/exdb/mnist/
     */
    private static function readLabels(string $labelPath): array
    {
        $stream = fopen($labelPath, 'rb');

        if (FALSE === $stream) {
            throw new \Exception('Could not open file: ' . $labelPath);
        }

        $labels = [];

        try {
            $header = fread($stream, 8);

            $fields = unpack('Nmagic/Nsize', $header);

            if ($fields['magic'] !== self::MAGIC_LABEL) {
                throw new \Exception('Invalid magic number: ' . $imagePath);
            }

            $labels = fread($stream, $fields['size']);
        } finally {
            fclose($stream);
        }

        return array_values(unpack('C*', $labels));
    }
}
