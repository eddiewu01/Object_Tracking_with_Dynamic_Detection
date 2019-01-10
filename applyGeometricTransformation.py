#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 21:19:57 2018

@author: edwardwu
"""
from skimage import transform
import numpy as np
import cv2

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):
    num_of_objs = startXs.shape[1]
    Xs = np.zeros([startXs.shape[0], num_of_objs]) - 1
    Ys = np.zeros([startXs.shape[0], num_of_objs]) - 1
    newbbox = np.zeros([num_of_objs, 4, 2])
    for i in range(num_of_objs):
        src_coords = np.column_stack((startXs[:,i], startYs[:,i]))
        dest_coords = np.column_stack((newXs[:,i], newYs[:,i]))

        tform = transform.estimate_transform('similarity', src_coords, dest_coords)

        tformp = np.asmatrix(tform.params)
        extra_column_ones = np.ones([src_coords.shape[0],1])

        src_coords = np.column_stack((src_coords, extra_column_ones))
        dest_coords = np.column_stack((dest_coords, extra_column_ones))
        new_coords = tformp.dot(src_coords.T)
        f = np.array(np.sqrt(np.sum(np.square(dest_coords-new_coords.T), axis = 1)) <= 5)
        f = f.reshape([f.shape[0],])
        
        
        src_coords = src_coords[f][:,:2]
        dest_coords = dest_coords[f][:,:2]
        tform2 = transform.estimate_transform('similarity', src_coords, dest_coords)
        tformp2 = np.asmatrix(tform2.params)
        
        Xs[:dest_coords.shape[0], i] = dest_coords[:,0] 
        Ys[:dest_coords.shape[0], i] = dest_coords[:,1] 
#        points = np.column_stack((Ys, Xs))
#        points = points[~(points==-1).all(1)] # remove rows with all 0s
        cur_bbox = bbox[i,:,:]
        cur_bbox = np.column_stack((cur_bbox, np.ones([cur_bbox.shape[0],1])))
        points = tformp2.dot(cur_bbox.T).T[:,:2]
        #points[:, [0, 1]] = points[:, [1,0]]
#        points = np.round(points).astype(int)
#        x,y,w,h = cv2.boundingRect(points)
        topx, bottomx = np.min(points[:,0]), np.max(points[:,0])
        lefty, righty = np.min(points[:,1]), np.max(points[:,1])
        #x,y,w,h = int(x),int(y),int(w),int(h)
        newbbox[i,:,:] = [[topx,lefty], [bottomx, lefty], [bottomx, righty], [topx, righty]]
#        newbbox[i,:,:] = [[y,x], [y+h, x], [y+h, x+w], [y, x+w]]

    f = ~(((Xs== -1).all(1)) & ((Ys== -1).all(1)))
    Xs, Ys = Xs[f], Ys[f]
    
    return Xs, Ys, newbbox
