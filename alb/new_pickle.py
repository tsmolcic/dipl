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
    dataset = np.ndarray(shape = (len(image_files), image_size, image_size, num_channels), dtype=np.float32)
    print(folder)

    count = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = imread(image_file).astype(float) / pixel_depth
            if image_data.shape != (image_size, image_size, num_channels):
                raise Exception('Unexpected image shape {}'.format(image_data.shape))
            dataset[count, :, :] = image_data
            count += 1
            if count % 500 == 0:
                sys.stdout.write("progress: {}   \r".format(count) )
                sys.stdout.flush()
        except(IOError, ValueError) as e:
            print('Couldn\'t load image {} - {} - skipping.'.format(image_file, e))
    
    print('Dataset shape: {}'.format(dataset.shape))
    print('Mean value of images: {:3f}'.format(np.mean(dataset)))
    print('Max: {:3f}'.format(np.max(dataset)))
    print('Min: {:3f}'.format(np.min(dataset)))
    print('Standard deviation of dataset: {:3f}'.format(np.std(dataset)))
    
    return dataset

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