import sys
import numpy as np
from skimage import io, transform
from glob import iglob
import pickle
# import pdb; pdb.set_trace()

def main():
    """
    if len(sys.argv) < 3:
        print("./crop.faces.py INPUT_DIR OUTPUT_DIR")
        return
    input_dir = sys.argv[1]
    """
    input_dir = "./DATA/CAT"
    result = "./get_face_coordinate.result"
    ret = []
    for i, image_path in enumerate(iglob('%s/*/*.jpg' % input_dir)):
        annotation_path = '%s.cat' % image_path
        try:
            annotation = parse_annotation(open(annotation_path).read())
        except:
            continue
        # face = crop_face(io.imread(image_path), annotation)
        face = get_coordinate(annotation, io.imread(image_path))
        if face == None:
            continue
        ret.append([image_path, face])
    pickle.dump(ret, open(result, 'wb'))

def parse_annotation(line):
    v = list(map(int, line.split()))
    ret = {}
    parts = ["left_eye", "right_eye", "mouth",
             "left_ear1", "left_ear2", "left_ear3",
             "right_ear1", "right_ear2", "right_ear3"]
    for i, part in enumerate(parts):
        if i >= v[0]: break
        ret[part] = np.array([v[1 + 2 * i], v[1 + 2 * i + 1]])
    return ret

def get_coordinate(an, image):
    diff_eyes = an["left_eye"] - an["right_eye"]
    if diff_eyes[0] == 0 or abs(float(diff_eyes[1]) / diff_eyes[0]) > 0.5:
        return None
    center = (an["left_eye"] + an["right_eye"] + an["mouth"]) / 3
    if center[1] > an["mouth"][1]: return None
    radius = np.linalg.norm(diff_eyes) * 1.1
    xu = int(center[0] - radius)
    xl = int(center[0] + radius)
    yu = int(center[1] - radius)
    yl = int(center[1] + radius)
    if xl > image.shape[1] or yl > image.shape[0]: return None
    if xu < 0 or yu < 0: return None
    face = {
        'x': xu,
        'y': yu,
        'width': radius * 2,
        'height':radius * 2}
    return face

if __name__ == "__main__":
    main()