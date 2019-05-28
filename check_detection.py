import sys
import pickle
from skimage import io, feature, color, transform
# import pdb; pdb.set_trace()

def main():
    detections = pickle.load(open("./get_detections.result", 'r+b'))
    faces = pickle.load(open("./get_face_coordinate.result", 'r+b'))
    correct = 0
    total = 0
    fit = get_fit(detections, faces)
    for i in fit:
        for j in i[2]:
            if j['fit'] >= 0.9:
                correct = correct + 1
        total = total + len(i[2])
    print("Accuracy:" + str(correct / total) + ", Correct:" + str(correct) + ", Total:" + str(total))
    pickle.dump(fit, open("./check_detection.result", 'wb'))

def get_fit(detections, faces):
    ret = []
    for i in detections:
        for j, k in enumerate(faces):
            try: hoge = k.index(i[1])
            except: continue
            break
        coordinates = []
        for l in i[2]:
            fit = overlap_score(l, faces[j][1])
            l.update({'fit' : fit})
            coordinates.append(l)
        ret.append([i[0], i[1], coordinates])
    return ret

def overlap_score(a, b):
    left = max(float(a['x']), float(b['x']))
    right = min(float(a['x']) + float(a['width']), float(b['x']) + float(b['width']))
    top = max(float(a['y']), float(b['y']))
    bottom = min(float(a['y']) + float(a['height']), float(b['y']) + float(b['height']))
    intersect = max(0, max(0, (right - left)) * max(0, (bottom - top)))
    union = float(a['width']) * float(a['height']) + float(b['width']) * float(b['height']) - intersect
    return intersect / union

if __name__ == "__main__":
    main()
