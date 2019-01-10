# -*- coding: utf-8 -*-

'''
load modules
'''
import os
import sys
import numpy as np
import skimage.io

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn import visualize


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR + '/Final_Project', "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR + '/Final_Project', "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

'''
Configurations
'''
class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80 # COCO has 80 classes

class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
config = InferenceConfig()
config.display()


def mrcnn_detect(img):
    '''
    Create and use pre-trained model
    '''
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    
    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']
    
    '''
    Object Detection
    '''
    results = model.detect([img], verbose=1)

    # Visualize results
    r = results[0]
#    print([class_names[cid] for cid in r['class_ids']]) # get object labels
    
    '''
    # write to pickle 
    with open('mrcnn_result.pkl', 'wb') as fo:
        pickle.dump(r['rois'], fo)
    '''
    # get closedly matched features
    sel_ind = np.where(r['scores'] > 0.95)[0]

    visualize.display_instances(img, r['rois'][sel_ind,:], r['masks'][:,:,sel_ind], r['class_ids'][sel_ind], 
                                class_names, r['scores'][sel_ind])
    
    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

    # return bbox and class names
    return r['rois'][sel_ind, :], [class_names[cid] for cid in r['class_ids'][sel_ind]]

