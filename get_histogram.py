"""
Load module
"""
import sys
import numpy as np
from skimage import io, feature

def main():
    if len(sys.argv) < 2:
        print ("./tmp1.py IMAGE_PATH")
        return
    image_path = sys.argv[1]
    histogram = get_histogram(io.imread(image_path, as_grey = True))
    feature_vector = histogram.reshape(-1)

"""
Calcurate feature as lbp and histogram
"""
def get_histogram(image):
    LBP_POINTS = 24
    LBP_RADIUS = 3
    CELL_SIZE = 8
    bins = LBP_POINTS + 2
    lbp = feature.local_binary_pattern(image, LBP_POINTS, LBP_RADIUS, 'uniform')
    histogram = np.zeros(shape = (int(image.shape[0] / CELL_SIZE), int(image.shape[1] / CELL_SIZE), bins), dtype = np.int)
    for y in range(0, int(image.shape[0] - CELL_SIZE), CELL_SIZE):
        for x in range(0, int(image.shape[1] - CELL_SIZE), CELL_SIZE):
            for dy in range(CELL_SIZE):
                for dx in range(CELL_SIZE):
                    histogram[int(y / CELL_SIZE), int(x / CELL_SIZE), int(lbp[y + dy, x + dx])] +=1
    return histogram

"""
This script will not be executed if called from another file
"""
if __name__ == "__main__":
    main()
