"""
Load module
"""
import sys
import pickle
from skimage import io, feature, color, transform
from get_detections import get_detections
# import pdb; pdb.set_trace()


def main():
    """
    svm = pickle.load(open(sys.argv[1], 'r+b'))
    target = color.rgb2gray(io.imread(sys.argv[2]))
    """
    svm = pickle.load(open("./make_model.result", 'r+b'))
    target = color.rgb2gray(io.imread("./DATA/CAT/CAT_00/00000001_000.jpg"))
    detections = get_detections(svm, target)
    detections_nms = apply_nms(detections)

"""
Calculate the overlap of two detection rectangles
"""
def overlap_score(a, b):
    left = max(float(a['x']), float(b['x']))
    right = min(float(a['x']) + float(a['width']), float(b['x']) + float(b['width']))
    top = max(float(a['y']), float(b['y']))
    bottom = min(float(a['y']) + float(a['height']), float(b['y']) + float(b['height']))
    intersect = max(0, max(0, (right - left)) * max(0, (bottom - top)))
    union = float(a['width']) * float(a['height']) + float(b['width']) * float(b['height']) - intersect
    return intersect / union

"""
Apply NMS
"""
def apply_nms(detections):
    detections = sorted(detections, key = lambda d: d['score'], reverse = True)
    deleted = set()
    for i in range(len(detections)):
        if i in deleted: continue
        for j in range(i + 1, len(detections)):
            if overlap_score(detections[i], detections[j]) > 0.3:
                deleted.add(j)
    detections = [d for i, d in enumerate(detections) if not i in deleted]
    return detections

"""
This script will not be executed if called from another file
"""
if __name__ == "__main__":
    main()
