import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


coords = []
def onclick(event):
    global ix, iy, coords
    ix, iy = event.xdata, event.ydata
    print('x = %d, y = %d'%(
        ix, iy))
    print("this is coords", coords)

    if coords and coords[-1] and len(coords[-1]) != 2:
        coords[-1].append([int(iy), int(ix)])
    else:
        coords.append([[int(iy), int(ix)]])
        
    return coords


def draw_bounding_box(img):
    fig = plt.figure()
    plt.imshow(img)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    

    res = np.zeros([len(coords), 4, 2])#, dtype = np.int)
    for obj in range(len(coords)):
        top_x, left_y = coords[obj][0][0], coords[obj][0][1]
        bottom_x, right_y = coords[obj][1][0], coords[obj][1][1]
        img_copy = img.copy()
        cv2.rectangle(img_copy, (left_y, top_x), (right_y, bottom_x), (0,255,0),2)
        plt.imshow(img_copy)
        plt.show()
        res[obj,:,:] = [[top_x, left_y], [bottom_x, left_y], [bottom_x, right_y], [top_x, right_y]]
      
    return res

