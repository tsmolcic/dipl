# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:32:36 2018

@author: Tomislav
"""

#import getch
from msvcrt import getch
import PIL
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from shutil import copy, move
import os
    
cwd = os.getcwd()

dirs = []

blobs_directory = cwd+'/sdss_data/blobs/'
valid_dir = cwd+'/sdss_data/valid'
notvalid_dir = cwd+'/sdss_data/notvalid'
potential_dir = cwd+'/sdss_data/potential/'
processed_dir = cwd+'/sdss_data/processed/'

dirs.append(blobs_directory)
dirs.append(valid_dir)
dirs.append(notvalid_dir)
dirs.append(potential_dir)
dirs.append(processed_dir)

for directory in dirs:
    if not os.path.exists(directory):
        os.makedirs(directory)

blobs = os.fsencode(blobs_directory)

valid = 0
invalid = 0
potential = 0

for file in os.listdir(blobs):
    filename = os.fsdecode(file)
    if (valid+potential+invalid) % 100 ==0:
        print('Total: {}\nValid: {}\nPotential: {}\nInvalid: {}\n'.format(valid + invalid + potential, valid, potential, invalid))
        print('Valid total: {}\n'.format(len(os.listdir(valid_dir)) - 1))
    if filename.endswith(".jpg"):
        print(filename)
        img_name = blobs_directory+filename
        img = mpimg.imread(img_name)
        imgplot = plt.imshow(img)
        plt.show()
        key = ord(getch())
        # plt.close()
#        print(key)
        if key == 3:
            print("\t\tAborted.\n\n")
            print('Total: {}\nValid: {}\nPotential: {}\nInvalid: {}\n'.format(valid + invalid + potential, valid, potential, invalid))
            print('Valid total: {}\n'.format(len(os.listdir(valid_dir)) - 1))
            break
        elif key == 99:
            copy(img_name, valid_dir)
            move(img_name, processed_dir)
            valid += 1
            print("\t\tPOSITIVE")
        elif key == 101:
            copy(img_name, potential_dir)
            move(img_name, processed_dir)
            potential += 1
            print("\t\tNeeds confirmation.")
        else:
            copy(img_name, notvalid_dir)
            move(img_name, processed_dir)
            invalid += 1
            print("\t\tNegative.")
    else:
        continue
    
