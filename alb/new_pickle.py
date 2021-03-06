import os
import sys
import numpy as np
from matplotlib.image import imread, imsave

from six.moves import cPickle as pickle

new_data_folder = ['./sdss_data/blobs']

image_size = 40 # 40x40 pix images
num_channels = 3 # .jpg
pixel_depth = 255

def load_image_folder(folder):
    image_files = os.listdir(folder)
    images1 = image_files[0:100000]
    images2 = image_files[100000:200000]
    images3 = image_files[200000:300000]
    images4 = image_files[300000:400000]
    images5 = image_files[500000:]
    dataset1 = np.ndarray(shape = (len(images1), image_size, image_size, num_channels), dtype=np.float32)
    dataset2 = np.ndarray(shape = (len(images2), image_size, image_size, num_channels), dtype=np.float32)
    dataset3 = np.ndarray(shape = (len(images3), image_size, image_size, num_channels), dtype=np.float32)
    dataset4 = np.ndarray(shape = (len(images4), image_size, image_size, num_channels), dtype=np.float32)
    dataset5 = np.ndarray(shape = (len(images5), image_size, image_size, num_channels), dtype=np.float32)
    print(folder)

    print("novi dio\n\n")
    count = 0
    for image in images1:
        image_file = os.path.join(folder, image)
        try:
            image_data = imread(image_file).astype(float) / pixel_depth
            if image_data.shape != (image_size, image_size, num_channels):
                raise Exception('Unexpected image shape {}'.format(image_data.shape))
            dataset1[count, :, :] = image_data
            count += 1
            if count % 500 == 0:
                sys.stdout.write("progress: {}   \r".format(count) )
                sys.stdout.flush()
        except(IOError, ValueError) as e:
            print('Couldn\'t load image {} - {} - skipping.'.format(image_file, e))
    
    print('Dataset shape: {}'.format(dataset1.shape))
    print('Mean value of images: {:3f}'.format(np.mean(dataset1)))
    print('Max: {:3f}'.format(np.max(dataset1)))
    print('Min: {:3f}'.format(np.min(dataset1)))
    print('Standard deviation of dataset: {:3f}'.format(np.std(dataset1)))

    print("novi dio\n\n")
    count = 0
    for image in images2:
        image_file = os.path.join(folder, image)
        try:
            image_data = imread(image_file).astype(float) / pixel_depth
            if image_data.shape != (image_size, image_size, num_channels):
                raise Exception('Unexpected image shape {}'.format(image_data.shape))
            dataset2[count, :, :] = image_data
            count += 1
            if count % 500 == 0:
                sys.stdout.write("progress: {}   \r".format(count) )
                sys.stdout.flush()
        except(IOError, ValueError) as e:
            print('Couldn\'t load image {} - {} - skipping.'.format(image_file, e))
    
    print('Dataset shape: {}'.format(dataset2.shape))
    print('Mean value of images: {:3f}'.format(np.mean(dataset2)))
    print('Max: {:3f}'.format(np.max(dataset2)))
    print('Min: {:3f}'.format(np.min(dataset2)))
    print('Standard deviation of dataset: {:3f}'.format(np.std(dataset2)))

    print("novi dio\n\n")
    count = 0
    for image in images3:
        image_file = os.path.join(folder, image)
        try:
            image_data = imread(image_file).astype(float) / pixel_depth
            if image_data.shape != (image_size, image_size, num_channels):
                raise Exception('Unexpected image shape {}'.format(image_data.shape))
            dataset3[count, :, :] = image_data
            count += 1
            if count % 500 == 0:
                sys.stdout.write("progress: {}   \r".format(count) )
                sys.stdout.flush()
        except(IOError, ValueError) as e:
            print('Couldn\'t load image {} - {} - skipping.'.format(image_file, e))
    
    print('Dataset shape: {}'.format(dataset3.shape))
    print('Mean value of images: {:3f}'.format(np.mean(dataset3)))
    print('Max: {:3f}'.format(np.max(dataset3)))
    print('Min: {:3f}'.format(np.min(dataset3)))
    print('Standard deviation of dataset: {:3f}'.format(np.std(dataset3)))

    print("novi dio\n\n")
    count = 0
    for image in images4:
        image_file = os.path.join(folder, image)
        try:
            image_data = imread(image_file).astype(float) / pixel_depth
            if image_data.shape != (image_size, image_size, num_channels):
                raise Exception('Unexpected image shape {}'.format(image_data.shape))
            dataset4[count, :, :] = image_data
            count += 1
            if count % 500 == 0:
                sys.stdout.write("progress: {}   \r".format(count) )
                sys.stdout.flush()
        except(IOError, ValueError) as e:
            print('Couldn\'t load image {} - {} - skipping.'.format(image_file, e))
    
    print('Dataset shape: {}'.format(dataset4.shape))
    print('Mean value of images: {:3f}'.format(np.mean(dataset4)))
    print('Max: {:3f}'.format(np.max(dataset4)))
    print('Min: {:3f}'.format(np.min(dataset4)))
    print('Standard deviation of dataset: {:3f}'.format(np.std(dataset4)))

    print("novi dio\n\n")
    count = 0
    for image in images5:
        image_file = os.path.join(folder, image)
        try:
            image_data = imread(image_file).astype(float) / pixel_depth
            if image_data.shape != (image_size, image_size, num_channels):
                raise Exception('Unexpected image shape {}'.format(image_data.shape))
            dataset5[count, :, :] = image_data
            count += 1
            if count % 500 == 0:
                sys.stdout.write("progress: {}   \r".format(count) )
                sys.stdout.flush()
        except(IOError, ValueError) as e:
            print('Couldn\'t load image {} - {} - skipping.'.format(image_file, e))
    
    print('Dataset shape: {}'.format(dataset5.shape))
    print('Mean value of images: {:3f}'.format(np.mean(dataset5)))
    print('Max: {:3f}'.format(np.max(dataset5)))
    print('Min: {:3f}'.format(np.min(dataset5)))
    print('Standard deviation of dataset: {:3f}'.format(np.std(dataset5)))
    
    return dataset1, dataset2, dataset3, dataset4, dataset5

def make_pickles(data_folders, force=False):
    dataset_names = []
    for dir in data_folders:
        filename = dir+'.pickle'
        dataset_names.append(filename)
        if os.path.exists(filename) and not force:
            print('{} already exists, skipping pickling.'.format(filename))
        else:
            print('Pickling {}...'.format(filename))
            dataset = load_image_folder(dir)
            try:
                with open(filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save to: {} - {}'.format(filename, e))

    return dataset_names

dataset = make_pickles(new_data_folder)