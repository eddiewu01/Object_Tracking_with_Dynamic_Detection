#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 08:28:12 2018

@author: cis581
"""


import numpy as np

from getFeatures import getFeatures
import cv2

from matplotlib import pyplot as plt
from interp2 import interp2

from scipy import signal

def transformFeatures(startX, startY):

  p0 = np.zeros((1,1,2),dtype = 'float32')
  p0[:, :, 0] = startY
  p0[:, :, 1] = startX
  return p0
  
def recoverFeaures(p0):
  startX = p0[:, :, 1]
  startY = p0[:, :, 0]
  
  return startX, startY


def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):
  
  im1 = img1.dot([0.587, 0.114, 0.299])
  im2 = img2.dot([0.587, 0.114, 0.299])
  
  
  # Parameters for lucas kanade optical flow
  lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
  p0 = transformFeatures(startX, startY)
  p1, st, err = cv2.calcOpticalFlowPyrLK(im1.astype('uint8'), im2.astype('uint8'), p0, None, **lk_params)
  
  
  oriX_opt, oriY_opt = recoverFeaures(p1)
  
  return oriX_opt, oriY_opt
  
  
  


  
  
  
  
  
  
  
  
  
  