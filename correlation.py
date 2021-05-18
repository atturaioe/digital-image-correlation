import numpy as np
from PIL import Image

import argparse


def standardize(arr):
    """Apply stadardization to the given array"""

    standardized = arr - np.mean(arr)
    standardized /= np.linalg.norm(standardized)  # L2 norm
    return standardized


def rescale(arr):
    """Min-max normalization. Rescale given array to values in range [0, 1]"""

    return (arr - arr.min()) / (arr.max() - arr.min())


def pad(temp_size):
    """Return the padding sizes relative to tem_size"""

    padding = (temp_size - 1) / 2
    if padding % 2 == 0:
        return int(padding), int(padding)
    else:
        return int(np.ceil(padding)), int(np.floor(padding))


def correlation2D(image, template):
    # TODO make rotated and scaled template sliding
    assert image.shape > template.shape, 'Template is bigger than image'

    img_h, img_w = image.shape
    temp_h, temp_w = template.shape
    temp = standardize(template)
    result = np.zeros(image.shape)

    pad_h, pad_w = pad(temp_h), pad(temp_w)

    image = np.pad(image, (pad_h, pad_w))

    for i in range(img_h):
        for j in range(img_w):
            left, right, top, bottom = j, j+temp_w, i, i+temp_h
            patch = image[top:bottom, left:right]
            assert patch.shape == temp.shape, 'Patch and and template should be the same size'
            patch = standardize(patch)
            result[i, j] = np.sum(np.multiply(patch, temp))

    return result

def parse_args():
    parser = argparse.ArgumentParser(description='2D grayscale cross-correlation')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--template', type=str, required=True)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    image = Image.open(args.image).convert('L')
    template = Image.open(args.template).convert('L')

    image = np.copy(np.asarray(image))
    template = np.copy(np.asarray(template))
    corr = correlation2D(image, template)

    output = Image.fromarray(np.uint8(rescale(corr) * 255))
    output.save('correlated.png', 'PNG')

if __name__ == '__main__':
    main()
