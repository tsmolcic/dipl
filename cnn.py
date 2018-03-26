import numpy as np
import tensorflow as tf

from six.moves import cPickle as pickle

from time import time

import scipy.misc

tf.logging.set_verbosity(tf.logging.ERROR)

t1 = time()

pickle_file = 'datasets3.pickle'
new_data_pickle_1 = './new_data/unknown.pickle'
new_data_pickle_2 = './new_data/blobs_0.pickle'
new_data_pickle_3 = './new_data/blobs_1.pickle'
new_data_pickle_4 = './new_data/blobs_2.pickle'

with open(pickle_file, 'rb') as f:
    temp = pickle.load(f)
    print([key for key, value in temp.items()])
    training_set = temp['training_set']
    training_set_labels = temp['training_set_labels']
    validation_set = temp['validation_set']
    validation_set_labels = temp['validation_set_labels']
    test_set = temp['test_set']
    test_set_labels = temp['test_set_labels']
    del temp
    f.close()

with open(new_data_pickle_1, 'rb') as f:
    new_data_1 = pickle.load(f)
    f.close()

with open(new_data_pickle_2, 'rb') as f:
    new_data_2 = pickle.load(f)
    f.close()

with open(new_data_pickle_3, 'rb') as f:
    new_data_3 = pickle.load(f)
    f.close()

with open(new_data_pickle_4, 'rb') as f:
    new_data_4 = pickle.load(f)
    f.close()    

#set variables
image_size = training_set.shape[1] #40
num_labels = len(np.unique(training_set_labels)) #2
num_channels = len(training_set.shape) - 1 #3

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

def reformat_new_data(dataset):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    return dataset

def accuracy(predictions, labels):
    return (100 * np.sum(np.argmax(predictions, 1) == np.argmax(labels,1)) / predictions.shape[0])

#datasets before reformating
print('\nTraining: {}, {}\nValidation: {}, {}\nTest: {}, {}'.format(training_set.shape, training_set_labels.shape, validation_set.shape, validation_set_labels.shape, test_set.shape, test_set_labels.shape))
#reformat datasets
training_set, training_set_labels = reformat(training_set, training_set_labels)
validation_set, validation_set_labels = reformat(validation_set, validation_set_labels)
test_set, test_set_labels = reformat(test_set, test_set_labels)
new_data_set_1 = reformat_new_data(new_data_1)
new_data_set_2 = reformat_new_data(new_data_2)
new_data_set_3 = reformat_new_data(new_data_3)
new_data_set_4 = reformat_new_data(new_data_4)
#datasets after reformating
print('\nTraining: {}, {}\nValidation: {}, {}\nTest: {}, {}'.format(training_set.shape, training_set_labels.shape, validation_set.shape, validation_set_labels.shape, test_set.shape, test_set_labels.shape))
print('New data: {}, {}, {}, {}'.format(new_data_set_1.shape, new_data_set_2.shape, new_data_set_3.shape, new_data_set_4.shape))
#### hyperparameters ###

batch_size = 64
patch_size = 5
depth = 64
num_hidden = 64

g = tf.Graph()

with tf.Session(graph = g) as sess:
    tf_training_dataset = tf.placeholder(tf.float32, shape = (batch_size, image_size, image_size, num_channels))
    tf_training_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
    tf_validation_dataset = tf.constant(validation_set)
    tf_test_dataset = tf.constant(test_set)
    tf_new_dataset_1 = tf.constant(new_data_set_1)
    tf_new_dataset_2 = tf.constant(new_data_set_2)
    tf_new_dataset_3 = tf.constant(new_data_set_3)
    tf_new_dataset_4 = tf.constant(new_data_set_4)
    
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))

    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

    layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    tf.global_variables_initializer().run()
    saver = tf.train.Saver()

    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        output = tf.matmul(hidden, layer4_weights) + layer4_biases
        return output
    
    logits = model(tf_training_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_training_labels, logits = logits))

    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    training_predictions = tf.nn.softmax(logits)
    validation_predictions = tf.nn.softmax(model(tf_validation_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))
    new_data_prediction_1 = tf.nn.softmax(model(tf_new_dataset_1))
    new_data_prediction_2 = tf.nn.softmax(model(tf_new_dataset_2))
    new_data_prediction_3 = tf.nn.softmax(model(tf_new_dataset_3))
    new_data_prediction_4 = tf.nn.softmax(model(tf_new_dataset_4))
    
    num_steps = 1001

    print('We have lifthoff!')
    for step in range(num_steps):
        offset = (step * batch_size) % (training_set_labels.shape[0] - batch_size)
        batch_data = training_set[offset : (offset + batch_size), :, :, :]
        batch_labels = training_set_labels[offset : (offset + batch_size)]

        feed_dict = {tf_training_dataset : batch_data, tf_training_labels : batch_labels}

        _, l, predictions = sess.run([optimizer, loss, training_predictions], feed_dict = feed_dict)
        if step % 50 == 0:
            print('Minibatch loss at step {}: {:2f}'.format(step, l))
            print('Minibatch accuracy: {:2f}'.format( accuracy(predictions, batch_labels) ))
            print('Validation accuracy: {:2f}\n\n'.format( accuracy(validation_predictions.eval(), validation_set_labels) ))

    predictions = test_prediction.eval()
    print('Test accuracy: {:2f}'.format( accuracy(predictions, test_set_labels) ))
    t2 = time()
    print('\n\nTraining time: {:2f} s'.format(t2-t1))
    save_path = saver.save(sess, './model/asteroid_model.ckpt')
    print('Model saved in path: {}'.format(save_path))

    new_predictions_1 = new_data_prediction_1.eval()
    new_predictions_1 = np.argmax(new_predictions_1,1)
    print(np.unique(new_predictions_1, return_counts = True))
    
    new_predictions_2 = new_data_prediction_2.eval()
    new_predictions_2 = np.argmax(new_predictions_2,1)
    print(np.unique(new_predictions_2, return_counts = True))

    new_predictions_3 = new_data_prediction_3.eval()
    new_predictions_3 = np.argmax(new_predictions_3,1)
    print(np.unique(new_predictions_3, return_counts = True))

    new_predictions_4 = new_data_prediction_4.eval()
    new_predictions_4 = np.argmax(new_predictions_4,1)
    print(np.unique(new_predictions_4, return_counts = True))
    # print(np.unique(np.argmax(new_predictions,1), return_counts = True))

ps = np.argmax(predictions, 1)
ts = np.argmax(test_set_labels, 1)

mistakes = []
for i in range(ps.shape[0]):
    if ps[i] != ts[i]:
        mistakes.append(i)
for num in mistakes:
    name = 'mistakes3/mistake'+str(num)+'.jpg'
    scipy.misc.imsave(name, test_set[num])

print('Mistakes on test set: {}\nSaved to mistakes directory.'.format(len(mistakes)))

count_1 = 0
for i in range(new_predictions_1.shape[0]):
    if new_predictions_1[i] == 1:
        name = 'maybe/c_1_'+str(i)+'.jpg'
        scipy.misc.imsave(name, new_data_1[i]*255)
        count_1 += 1

print('Possible asteroids found on new dataset: {}\nSaved to maybe directory.'.format(count_1))

count_2 = 0
for i in range(new_predictions_2.shape[0]):
    if new_predictions_2[i] == 1:
        name = 'maybe/c_2_'+str(i)+'.jpg'
        scipy.misc.imsave(name, new_data_2[i]*255)
        count_2 += 1

print('Possible asteroids found on new dataset: {}\nSaved to maybe directory.'.format(count_2))

count_3 = 0
for i in range(new_predictions_3.shape[0]):
    if new_predictions_3[i] == 1:
        name = 'maybe/c_3_'+str(i)+'.jpg'
        scipy.misc.imsave(name, new_data_3[i]*255)
        count_3 += 1

print('Possible asteroids found on new dataset: {}\nSaved to maybe directory.'.format(count_3))

count_4 = 0
for i in range(new_predictions_4.shape[0]):
    if new_predictions_4[i] == 1:
        name = 'maybe/c_4_'+str(i)+'.jpg'
        scipy.misc.imsave(name, new_data_4[i]*255)
        count_1 += 1

print('Possible asteroids found on new dataset: {}\nSaved to maybe directory.'.format(count_4))