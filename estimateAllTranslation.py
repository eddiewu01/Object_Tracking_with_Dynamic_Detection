#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 23:48:56 2018

@author: cis581
"""
import numpy as np

from getFeatures import getFeatures
import cv2

from matplotlib import pyplot as plt
from interp2 import interp2

from scipy import signal

from estimateFeatureTranslation import estimateFeatureTranslation



def transformFeatures(startX, startY):

  valid = startX > 0 
  validXs = startX[valid]
  validYs = startY[valid]
  n = len(validXs)
  
  p0 = np.zeros((n,1,2),dtype = 'float32')
  p0[:, :, 0] = validYs.reshape(-1, 1)
  p0[:, :, 1] = validXs.reshape(-1, 1)
  
  return p0
  
def recoverFeaures(p0):
  startX = p0[:, :, 1].transpose()[0]
  startY = p0[:, :, 0].transpose()[0]
  
  return startX, startY

def estimateAllTranslation(startXs, startYs, img1, img2):
  
  im1 = img1.dot([0.587, 0.114, 0.299])
  im2 = img2.dot([0.587, 0.114, 0.299])

  n,f = startXs.shape
  newXs = np.zeros((n,f)) - 1
  newYs = np.zeros((n,f)) - 1
  
  
  
  for i in range(f):
  
    startX, startY = startXs[:, i], startYs[:, i]
    p0 = transformFeatures(startX, startY)
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    p1, st, err = cv2.calcOpticalFlowPyrLK(im1.astype('uint8'), im2.astype('uint8'), p0, None, **lk_params)
    numValid = 0
    if p1 is not None: 
      numValid = len(p1)
    
      newX, newY = recoverFeaures(p1)
      newXs[:, i][:numValid] = newX
      newYs[:, i][:numValid] = newY
      
    
  return newXs, newYs
    
    
    
  
  
  