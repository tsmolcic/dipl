# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:55:29 2018

@author: Tomislav
"""

import os
from math import sqrt

from imageio import imread, imwrite
from skimage.color import rgb2gray
from skimage.feature import blob_log

num_of_raw_img_dirs = 12

cwd = os.getcwd()

blobs_dir = cwd+'/sdss_data/blobs'

if not os.path.exists(blobs_dir):
    os.makedirs(blobs_dir)

img_dirs = []

print('creating directories...\n')
for i in range(num_of_raw_img_dirs):
    dir_name = cwd+'/sdss_data/raw_imgs_'+str(i)
    img_dirs.append(dir_name)

num = 0
for directory in img_dirs:
    images = os.fsencode(directory)
    for file in os.listdir(images):
        filename = os.fsdecode(file)
        if filename.endswith('.jpg'):
            print(filename)
            blobs_count = 0
            try:
                img = imread(img_dirs[num]+'/'+filename)    
                img_gray = rgb2gray(img)
                
                blobs_log = blob_log(img_gray, min_sigma=5, max_sigma=30, num_sigma=10, threshold=.075)
                blobs_log[:,2] = blobs_log[:,2] * sqrt(2)
                
                X = blobs_log[:,1]
                Y = blobs_log[:,0]
                
                fileName2 = filename.replace('.jpg', '')
                
                for i in range(X.shape[0]):
                    y1 = int(Y[i]-20)
                    y2 = int(Y[i]+20)
                    x1 = int(X[i]-20)
                    x2 = int(X[i]+20)
                    if y1 < 0 or x1 < 0 or y2 > img.shape[0] or x2 > img.shape[1]:
                        continue
                    else:
                        t = img[y1:y2 , x1:x2]
                        blobname = fileName2+'blob'+str(i)+'.jpg'
                        imwrite(blobs_dir+'/'+blobname, t)
                        blobs_count += 1
                print('num of blobs: '+str(blobs_count))
            except Exception as e:
                print('Unexpected file format... skipping.')
    num += 1