"""
Load module
"""
import pickle
import sys
from skimage import io, feature, color, transform
from get_histogram import get_histogram
from glob import iglob
import time
# import pdb; pdb.set_trace()

def main():
    start = time.time()
    # svm = pickle.load(open("./make_model.result", 'r+b'))
    svm = pickle.load(open("./DATA/RESULT/make_model.result" , 'r+b'))
    # target_dir = "./DATA/CAT/CAT_00"
    target_dir = "./DATA/tmp_cat"
    results = []
    for i, image_path in enumerate(iglob('%s/*.jpg' % target_dir)):
        results.append([i, image_path])
        target = color.rgb2gray(io.imread(image_path))
        detections = get_detections(svm, target)
        results[i].append(detections)
        elapsed_time = time.time() - start
        print (str(i + 1) + "/:1706 " + "elapsed_time:{0}".format(elapsed_time) + "[sec]")
    pickle.dump(results, open("./get_detections.result", 'wb'))

"""
Detection Proc
"""
def get_detections(svm, target):
    # Detection window size, must be same train image size
    WIDTH, HEIGHT = (64, 64)
    CELL_SIZE = 8
    THRESHOLD = 3.0
    target_scaled = target + 0
    scale_factor = 2.0 ** (-1.0 / 8.0)
    # Set result list
    detections = []
    for s in range(16):
        # Get image histogram
        histogram = get_histogram(target_scaled)
        # Check detction window with sliding  cell size
        for y in range(0, histogram.shape[0] - int(HEIGHT / CELL_SIZE)):
            for x in range(0, histogram.shape[1] - int(WIDTH / CELL_SIZE)):
                # Get feature vector and score
                feature = histogram[y:(y + int(HEIGHT / CELL_SIZE)), x:(x + int(WIDTH / CELL_SIZE))].reshape(1, -1)
                # score = svm.decision_function(feature)
                score = svm.predict(feature)
                # if score[0] > THRESHOLD:
                if score[0] == 1:
                    scale = (scale_factor ** s)
                    detections.append({
                        'x': x * CELL_SIZE / scale,
                        'y': y * CELL_SIZE / scale,
                        'width': WIDTH / scale,
                        'height': HEIGHT / scale,
                        'score': score})
        target_scaled = transform.rescale(target_scaled, scale_factor)
    return detections

"""
This script will not be executed if called from another file
"""
if __name__ == "__main__":
    main()
