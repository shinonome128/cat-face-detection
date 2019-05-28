import sys
import pickle
from glob import iglob
import numpy as np
from skimage import io, transform
# import pdb; pdb.set_trace()

def main():
    output_dir = "./DATA/NEGATIVE_ADD"
    fit = pickle.load(open("./check_detection.result", 'r+b'))
    for i in fit:
        for j, k in enumerate(i[2]):
            if k['fit'] < 9.0:
                negative = crop_negative(io.imread(i[1]), k)
                io.imsave('%s/%d_%d.png' % (output_dir,i[0],j), negative)

def crop_negative(image,l):
    x = int(l['x'])
    y = int(l['y'])
    width = int(l['width'])
    height = int(l['height'])
    cropped = image[y:(y + height), x:(x + width)]
    return transform.resize(cropped, (64, 64))

if __name__ == "__main__":
    main()
