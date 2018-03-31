import os
import sys

import numpy as np
from matplotlib.image import imread, imsave

from six.moves import cPickle as pickle

new_data_folder = './sdss_data/blobs'

image_size = 40 # 40x40 pix images
num_channels = 3 # .jpg
pixel_depth = 255

num_per_pickle = 10000

image_files = os.listdir(new_data_folder)

def make_pickle(k, force = False):
    start = k * num_per_pickle
    end = start + num_per_pickle
    if end > len(image_files):
        end = len(image_files)

    images = image_files[start : end]
    dataset = np.ndarray( shape = (num_per_pickle, image_size, image_size, num_channels), dtype = np.float32)

    n = 0
    for image in images:
        image_file = os.path.join(new_data_folder, image)
        try:
            image_data = imread(image_file).astype(float) / pixel_depth
            if image_data.shape != (image_size, image_size, num_channels):
                raise Exception('Unexpected image shape {}'.format(image_data.shape))
            dataset[n, :, :] = image_data
            n += 1
            if n % 500 == 0:
                sys.stdout.write("progress: {}/{}   \r".format(n, num_per_pickle) )
                sys.stdout.flush()
        except(IOError, ValueError) as e:
            print('Couldn\'t load image {} - {} - skipping.'.format(image_file, e))

    print('Dataset shape: {}'.format(dataset.shape))
    print('Mean value of images: {:3f}'.format(np.mean(dataset)))
    print('Max: {:3f}'.format(np.max(dataset)))
    print('Min: {:3f}'.format(np.min(dataset)))
    print('Standard deviation of dataset: {:3f}'.format(np.std(dataset)))

    filename = 'blobs'+str(k)+'.pickle'

    if os.path.exists(filename) and not force:
        print('{} already exists, skipping pickling.'.format(filename))
    else:
        print('Pickling {}...'.format(filename))
        try:
            with open(filename, 'wb') as f:
                pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
                f.close()
        except Exception as e:
            print('Unable to save to: {} - {}'.format(filename, e))

num_of_pickles = (len(image_files) // num_per_pickle) + 1

for i in range(num_of_pickles):
    print('Creating pickle {}/{}...'.format(i, num_of_pickles))
    make_pickle(i)