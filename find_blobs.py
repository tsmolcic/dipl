# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:55:29 2018

@author: Tomislav
"""

from math import sqrt
from skimage.feature import blob_log
from skimage.color import rgb2gray
from imageio import imread, imwrite
import os

cwd = os.getcwd()
img_dir = cwd+'/sdss_data/raw_imgs'
blobs_dir = cwd+'/sdss_data/blobs/'

dirs = []

dirs.append(img_dir)
dirs.append(blobs_dir)

for directory in dirs:
    if not os.path.exists(directory):
        os.makedirs(directory)

images = os.fsencode(img_dir)

num = 1
for file in os.listdir(images):
    filename = os.fsdecode(file)
    if filename.endswith('.jpg'):
        print(filename)
        img = imread(img_dir+'/'+filename)
        img_gray = rgb2gray(img)
        
        blobs_log = blob_log(img_gray, min_sigma=5, max_sigma=30, num_sigma=2, threshold=.1)
        blobs_log[:,2] = blobs_log[:,2] * sqrt(2)
        
        X = blobs_log[:,1]
        Y = blobs_log[:,0]
        
        for i in range(X.shape[0]):
            y1 = int(Y[i]-20)
            y2 = int(Y[i]+20)
            x1 = int(X[i]-20)
            x2 = int(X[i]+20)
            if y1 < 0 or x1 < 0 or y2 > img.shape[0] or x2 > img.shape[1]:
                continue
            else:
                t = img[y1:y2 , x1:x2]
                blobname = 'img'+str(num)+'blob'+str(i)+'.jpg'
                imwrite(blobs_dir+'/'+blobname, t)
    num += 1