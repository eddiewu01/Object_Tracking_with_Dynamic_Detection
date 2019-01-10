#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 11:52:57 2018

@author: cis581
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
from estimateFeatureTranslation import estimateFeatureTranslation
from estimateAllTranslation import estimateAllTranslation
from parseVideo import get_video_as_numpy
from getFeatures import getFeatures
from applyGeometricTransformation import applyGeometricTransformation
from draw_bounding_box import draw_bounding_box
from mrcnn_detect import InferenceConfig, mrcnn_detect

def reject_outliers(data, m=2):
    return abs(data - np.mean(data)) < m * np.std(data)


def outOfBounds(bbox, frame):
  inBound = True
  h, w = frame.shape
  x_low, y_low, x_high, y_high = bbox[0, 0], bbox[0, 1], bbox[2, 0], bbox[2, 1]
  
  if x_low < 0 or x_high > h - 1:

    inBound = False
  if y_low < 0 or y_high > w - 1:
    inBound = False
 

  return not inBound
  

def bgdMask(bboxs, frame):
   h, w = frame.shape
   mask = np.zeros_like(frame, dtype = bool)
   for obj in bboxs:
     if np.isnan(obj).any():
        continue
     x_low, y_low, x_high, y_high = obj[0, 0], obj[0, 1], obj[2, 0], obj[2, 1]
     
     ys, xs = np.meshgrid(np.arange(y_low, y_high), np.arange(x_low, x_high))
     xs = np.clip(xs , 0, h - 1)
     ys = np.clip(ys , 0, w - 1)
     mask[xs.flatten().astype(int), ys.flatten().astype(int)] = True
     
   return ~mask
 
  
def IsMoving(p0, p1, bboxs, new_bboxs, thre, frame):
  h, w = frame.shape
  thre = 0.002
  oldBgXs, oldBgYs = p0[:,  1].flatten(), p0[:,  0].flatten()
  newBgXs, newBgYs = p1[:,  1].flatten(), p1[:,  0].flatten()
  
  f = len(bboxs)
  movingBox = np.zeros((f,), dtype = bool)
  
  bg_dx, bg_dy = newBgXs - oldBgXs, newBgYs - oldBgYs
  xValid = reject_outliers(bg_dx)
  yValid = reject_outliers(bg_dx)

  bg_dx = bg_dx[xValid & yValid]

  bg_dy = bg_dy[xValid & yValid]

  bg_d = np.array([np.mean(bg_dx), np.mean(bg_dy)])
  
  for i in range(f):
    

    
    center_d = np.mean(new_bboxs[i], axis = 0) - np.mean(bboxs[i], axis = 0)

    if np.linalg.norm((center_d - bg_d) / (h, w)) > thre:
      
      movingBox[i] = True
      
      
  return movingBox

    

    
  

def objectTracking(rawVideo):
  
  
  video_arr = get_video_as_numpy(rawVideo)
 
  
  
  img1 = video_arr[0]
  img1_grey  = img1.dot([0.299, 0.587, 0.114])
  h , w = img1_grey.shape

  raw_bbox, class_names = mrcnn_detect(img1)
  
  bbox = np.zeros((len(raw_bbox), 4 ,2), dtype = int)
  for k in range(len(raw_bbox)):
    x_low, y_low, x_high, y_high = raw_bbox[k]
    bbox[k, 0, :] = (x_low, y_low)
    bbox[k, 1, :] = (x_high, y_low)
    bbox[k, 2, :] = (x_high, y_high)
    bbox[k, 3, :] = (x_low, y_high)
  
  # params for ShiTomasi corner detection
  feature_params = dict( maxCorners = 500,
                         qualityLevel = 0.0001,
                         minDistance = 3,
                         blockSize = 3 )
  # Parameters for lucas kanade optical flowimg1_grey
  lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
  p0 = cv2.goodFeaturesToTrack(img1_grey.astype('uint8'), mask = None, **feature_params)

  bg_mask = bgdMask(bbox, img1_grey)
  plt.figure()
  plt.imshow(bg_mask)
  plt.show()
  
  
  pxs = np.clip(p0[:, :, 1], 0, h - 1).astype(int)
  pys = np.clip(p0[:, :, 0], 0, w - 1).astype(int)
  
  is_bg = bg_mask[pxs, pys]
  bg_p0 = p0[is_bg].reshape(-1 ,1, 2)


  bbox_h_threshold = abs(bbox[0][1][0] - bbox[0][0][0]) * 2.5
  bbox_w_threshold = abs(bbox[0][3][1] - bbox[0][0][1]) * 2.5
  
  startXs, startYs = getFeatures(img1_grey, bbox)
  
  starting_num_features = startXs.shape[0]
  
  oldXs, oldYs = startXs, startYs
  oldFrame = img1.copy()
  for obj in bbox:

    x1, y1 = int(np.round(obj[0][1])), int(np.round(obj[0][0]))
    x2, y2 = int(np.round(obj[2][1])), int(np.round(obj[2][0]))
    cv2.rectangle(img1, (x1, y1),(x2, y2),(255, 150, 150), 1)
  n, f = startXs.shape
  for i in range(n):
    for j in range(f):
      if startXs[i][j] == -1:
        continue
  
  newVideoName = 'output_videos/' + rawVideo.split('/')[1].split('.')[-2] + '_result' + '.avi'
  

  _, h, w, _ = video_arr.shape
  writer = cv2.VideoWriter(newVideoName, cv2.VideoWriter_fourcc('M','J','P','G'), 25, (w, h), isColor = True)
  writer.write(img1[:, :, [2, 1, 0]])
  traj_x = []
  traj_y = []
  
  for idx in range(1, len(video_arr)):
    newFrame = video_arr[idx]
    newXs, newYs = estimateAllTranslation(oldXs, oldYs, oldFrame, newFrame)
    new_grey = newFrame.dot([0.299, 0.587, 0.114])
    old_gray = oldFrame.dot([0.299, 0.587, 0.114])

    
    bg_p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray.astype('uint8'), new_grey.astype('uint8'), bg_p0, None, **lk_params)
 
    
    restXs, restYs, newbbox = applyGeometricTransformation(oldXs, oldYs, newXs, newYs, bbox)
   
    if len(bg_p0) == 0:
        break
    movingBox = IsMoving(bg_p0[st == 1], bg_p1[st == 1], bbox, newbbox, 0.5, new_grey)
    
    
    
    traj_x += list(restXs.flatten())
    traj_y += list(restYs.flatten())
    if len(restXs) < 2:
        break
    

    bg_mask = bgdMask(newbbox, new_grey)
    p0 = cv2.goodFeaturesToTrack(new_grey.astype('uint8'), mask = None, **feature_params)
    
    try:
        pxs = np.clip(p0[:, :, 1], 0, h - 1).astype(int)
        pys = np.clip(p0[:, :, 0], 0, w - 1).astype(int)

        is_bg = bg_mask[pxs, pys]
        bg_p0 = p0[is_bg].reshape(-1, 1, 2)
    except:
        break
   
    
    oldFrame = newFrame.copy()
    oldXs, oldYs = restXs, restYs

    bbox = newbbox.copy()
    old_class_names = class_names
    if idx % 50 == 0: 
        raw_bbox, class_names = mrcnn_detect(video_arr[idx])
        bbox = np.zeros((len(raw_bbox), 4 ,2), dtype = int)
        for k in range(len(raw_bbox)):
            x_low, y_low, x_high, y_high = raw_bbox[k]
            bbox[k, 0, :] = (x_low, y_low)
            bbox[k, 1, :] = (x_high, y_low)
            bbox[k, 2, :] = (x_high, y_high)
            bbox[k, 3, :] = (x_low, y_high)
        oldXs, oldYs = getFeatures(new_grey, bbox)    

    for i in range(len(newbbox)):
      obj = newbbox[i]

      if np.isnan(obj).any():
        continue
      if outOfBounds(obj, new_grey):

        continue
    

      x1, y1 = int(np.round(obj[0][1])), int(np.round(obj[0][0]))
      x2, y2 = int(np.round(obj[2][1])), int(np.round(obj[2][0]))
      
      font = cv2.FONT_HERSHEY_SIMPLEX
      bottomLeftCornerOfText = ((x1, y1))
      fontScale = 0.6
      lineType = 2
      
      if movingBox[i]:
        cv2.rectangle(newFrame, (x1, y1),(x2, y2),(255, 150, 150), 2)
        cv2.putText(newFrame,'moving ' + old_class_names[i], 
            bottomLeftCornerOfText, 
            font,
            fontScale,
            (255, 150, 150),
            lineType)
      else:
        cv2.rectangle(newFrame, (x1, y1),(x2, y2),(0, 150, 150), 2)
        cv2.putText(newFrame,'still ' + old_class_names[i], 
            bottomLeftCornerOfText, 
            font,
            fontScale,
            (0, 150, 150),
            lineType)
      
        
    n, f = oldXs.shape
    
    writer.write(newFrame[:, :, [2, 1, 0]])
    
  return newVideoName