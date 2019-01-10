# Object Tracking with Dynamic Detection in Sliding Camera 
# Team Members: Youjia Li, Eddie Wu

&nbsp;
## Structure

#### objectTracking
`objectTracking.py` contains all the essential parts of optical flow operation, including `getFeatures`,`estimateAllTranslation` and `applyGeometricTransformation.py`.

#### mrcnn_detect
`mrcnn_detect.py` contains all the essential parts of object detection and instance segmentation, making use of published Mask R-CNN architecture, and it is used in `objectTracking.py`

##### create_output_video
`create_output_video.py` runs everything together through calling of `objectTracking` function. 

#### input_videos
This directory contains all the input videos we are testing. 

#### output_videos
This directory contains all the resulting output videos we are generating for this project.

&nbsp;
## Development Environment & Usage
1. Unzip the zipped file
2. Change directory to the `Final_Project` folder using cd in command line
3. (Optional) Create a virtual environment using `virtualenv venv --python=python3.6` 
4. Run `pip install -r requirements.txt`. If using CPU only, install CPU-Optimized Tensorflow, and make sure Python version is 3.6 or less (if virtual environment was not created per last step, as 3.7 doesn't fully support Tensorflow yet)
5. Modify the `rawVideo` variable in `create_output_video.py` to be the name of input video file
6. Run `create_output_video.py`. If `mask_rcnn_coco.h5` (pretrained Mask R-CNN weights) is not downloaded yet, it will first be downloaded to the current directory and then used for the application.
