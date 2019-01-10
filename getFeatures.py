#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 14:15:33 2018

@author: edwardwu
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def getFeatures(img, bbox):
    y, x = [], []
    N = -float('inf')
    num_of_objs = bbox.shape[0]
    for i in range(num_of_objs):
        topx, bottomx = bbox[i, 0, 0], bbox[i, 1, 0]
        lefty, righty = bbox[i, 0, 1], bbox[i, 2, 1]
        obj = img[int(topx):int(bottomx+1), int(lefty):int(righty+1)]
        obj = np.float32(obj)
        corners = cv2.goodFeaturesToTrack(obj, maxCorners = 100, qualityLevel = 0.001, minDistance = 3)
        corners = corners[:,0,:]
        corners[:, 0] += topx
        corners[:, 1] += lefty
        corners = corners[(corners[:, 0] <= bottomx) & (corners[:, 0] >= topx) & (corners[:,1] <= righty) & (corners[:,1] >= lefty)]

        N = max(N, corners.shape[0])
        y.append(corners[:, 0])
        x.append(corners[:, 1])
        
    res_y, res_x = np.zeros([N, num_of_objs]), np.zeros([N, num_of_objs]) - 1
    for i in range(num_of_objs):
        res_y[:len(y[i]), i] = y[i]
        res_x[:len(x[i]), i] = x[i]
    return res_y, res_x
        
    
    
    
        
    
    