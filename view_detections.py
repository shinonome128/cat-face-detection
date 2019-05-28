"""
Load module
"""
import sys
import pickle
from skimage import io, feature, color, transform
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from get_detections import get_detections
from apply_nms import apply_nms
# import pdb; pdb.set_trace()

check = []

def main():
    results = pickle.load(open("./get_detections.result", 'r+b'))
    new_negatives = []
    for i in results:
        target = color.rgb2gray(io.imread(i[1]))
        detections = i[2]
        detections = apply_nms(detections)
        check_detections(target, detections)
        if check[-1] == 'n':
            new_negatives.append([i[1], i[2]])
    pickle.dump(new_negatives, open("./view_detections.result", 'wb'))

"""
Display results
"""
def check_detections(target, detections):
    fig, (ax1) = plt.subplots(ncols = 1, figsize = (10, 10))
    ax1.imshow(target, cmap = cm.Greys_r)
    ax1.set_axis_off()
    ax1.set_title('result')
    for i in detections:
        y = int(i['y'])
        x = int(i['x'])
        tw = int(i['width'])
        th = int(i['height'])
        rect = plt.Rectangle((y, x),
                             tw,
                             th,
                             edgecolor = 'w',
                             facecolor = 'none',
                             linewidth = 2.5)
        ax1.add_patch(rect)
    cid = fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()

def onkey(event):
    global check
    if event.key == 'n':
        check.append(event.key)
        plt.close(event.canvas.figure)
    else:
        check.append('o')
        plt.close(event.canvas.figure)

"""
This script will not be executed if called from another file
"""
if __name__ == "__main__":
    main()
