from __future__ import absolute_import, division, print_function

import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
import tensorflow as tf
import numpy as np

from models import text_objseg_model as segmodel
from util import data_reader
from util import loss

# Parameters
# Model Params
T = 20
N = 50
vocab_size = 8803
embedded_dim = 1000
lstm_dim = 1000
mlp_hidden_dims = 500

# Initialization Params
convnet_params = './models/convert_caffemodel/params/vgg_params.npz'
mlp_l1_std = 0.05
mlp_l2_std = 0.1

# Training Params
positive_loss_multiplier = 1.
negative_loss_multiplier = 1.

start_learningrate = 0.01
learningrate_decay_step = 10000
learningrate_decay_rate = 0.1
weight_decay = 0.005
momentum = 0.8
max_iter = 25000

fix_convnet = True
vgg_dropout = False
mlp_dropout = False
vgg_learningrate_mult = 1.

# Data Params
data_folder = './exp-referit/data/train_batch_det/'
data_prefix = 'referit_train_det'

# Snapshot Params
snapshot = 5000
snapshot_file = './exp-referit/tfmodel/referit_fc8_detect_iteration_%d.tfmodel'

# The model
# Inputs
text_seq_batch = tf.placeholder(tf.int32, [T, N])
imagecrop_batch = tf.placeholder(tf.float32, [N, 224, 224, 3])
spatial_batch = tf.placeholder(tf.float32, [N, 8])
label_batch = tf.placeholder(tf.float32, [N, 1])

# Outputs
scores = segmodel.text_objseg_region(text_seq_batch, imagecrop_batch,
    spatial_batch, vocab_size, embedded_dim, lstm_dim, mlp_hidden_dims,
    vgg_dropout=vgg_dropout, mlp_dropout=mlp_dropout)

# Collect trainable variables, regularized variables and learning rates
# Only train the fc layers of convnet and keep conv layers fixed
if fix_convnet:
    train_variable_list = [variable for variable in tf.trainable_variables()
                      if not variable.name.startswith('vgg_local/')]
else:
    train_variable_list = [variable for variable in tf.trainable_variables()
                      if not variable.name.startswith('vgg_local/conv')]
print('Collecting variables to train:')
for variable in train_variable_list: print('\t%s' % variable.name)
print('Done.')

# Add regularization to weight matrices (excluding bias)
regularization_variable_list = [variable for variable in tf.trainable_variables()
                if (variable in train_variable_list) and
                (variable.name[-9:-2] == 'weights' or variable.name[-8:-2] == 'Matrix')]
print('Collecting variables for regularization:')
for variable in regularization_variable_list: print('\t%s' % variable.name)
print('Done.')

# Collect learning rate for trainable variables
variable_learningrate_mult = {variable: (vgg_learningrate_mult if variable.name.startswith('vgg_local') else 1.0)
               for variable in train_variable_list}
print('Variable learning rate multiplication:')
for variable in train_variable_list:
    print('\t%s: %f' % (variable.name, variable_learningrate_mult[variable]))
print('Done.')

# Loss function and accuracy
classification_loss = loss.weighed_logistic_loss(scores, label_batch, positive_loss_multiplier, negative_loss_multiplier)
regularization_loss = loss.l2_regularization_loss(regularization_variable_list, weight_decay)
total_loss = classification_loss + regularization_loss

def compute_accuracy(scores, labels):
    is_positive = (labels != 0)
    is_negative = np.logical_not(is_positive)
    num_all = labels.shape[0]
    num_positive = np.sum(is_positive)
    num_negative = num_all - num_positive

    is_correct = np.logical_xor(scores < 0, is_positive)
    accuracy_all = np.sum(is_correct) / num_all
    accuracy_positive = np.sum(is_correct[is_positive]) / num_positive
    accuracy_negative = np.sum(is_correct[is_negative]) / num_negative
    return accuracy_all, accuracy_positive, accuracy_negative

# Solver
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(start_learningrate, global_step, learningrate_decay_step,
    learningrate_decay_rate, staircase=True)
solver = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
# Compute gradients
grads_and_variables = solver.compute_gradients(total_loss, variable_list=train_variable_list)
# Apply learning rate multiplication to gradients
grads_and_variables = [((g if variable_learningrate_mult[v] == 1 else tf.mul(variable_learningrate_mult[v], g)), v)
                  for g, v in grads_and_variables]
# Apply gradients
train_step = solver.apply_gradients(grads_and_variables, global_step=global_step)

# Initialize parameters and load data
init_ops = []
# Initialize CNN Parameters
convnet_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
                  'conv3_1', 'conv3_2', 'conv3_3',
                  'conv4_1', 'conv4_2', 'conv4_3',
                  'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']
processed_params = np.load(convnet_params)
processed_W = processed_params['processed_W'][()]
processed_B = processed_params['processed_B'][()]
with tf.variable_scope('vgg_local', reuse=True):
    for l_name in convnet_layers:
        assign_W = tf.assign(tf.get_variable(l_name + '/weights'), processed_W[l_name])
        assign_B = tf.assign(tf.get_variable(l_name + '/biases'), processed_B[l_name])
        init_ops += [assign_W, assign_B]

# Initialize classifier Parameters
with tf.variable_scope('classifier', reuse=True):
    mlp_l1 = tf.get_variable('mlp_l1/weights')
    mlp_l2 = tf.get_variable('mlp_l2/weights')
    init_mlp_l1 = tf.assign(mlp_l1, np.random.normal(
        0, mlp_l1_std, mlp_l1.get_shape().as_list()).astype(np.float32))
    init_mlp_l2 = tf.assign(mlp_l2, np.random.normal(
        0, mlp_l2_std, mlp_l2.get_shape().as_list()).astype(np.float32))

init_ops += [init_mlp_l1, init_mlp_l2]
processed_params.close()

# Load data
reader = data_reader.DataReader(data_folder, data_prefix)

snapshot_saver = tf.train.Saver()
sess = tf.Session()

# Run Initialization operations
sess.run(tf.initialize_all_variables())
sess.run(tf.group(*init_ops))

# Optimization loop
classification_loss_avg = 0
avg_accuracy_all, avg_accuracy_positive, avg_accuracy_negative = 0, 0, 0
decay = 0.99

# Run optimization
for n_iter in range(max_iter):
    # Read one batch
    batch = reader.read_batch()
    text_seq_val = batch['text_seq_batch']
    imagecrop_val = batch['imagecrop_batch'].astype(np.float32) - segmodel.vgg_net.channel_mean
    spatial_batch_val = batch['spatial_batch']
    label_val = batch['label_batch'].astype(np.float32)

    loss_mult_val = label_val * (positive_loss_multiplier - negative_loss_multiplier) + negative_loss_multiplier

    # Forward and Backward pass
    scores_val, classification_loss_val, _, learningrate_val = sess.run([scores, classification_loss, train_step, learning_rate],
        feed_dict={
            text_seq_batch  : text_seq_val,
            imagecrop_batch    : imagecrop_val,
            spatial_batch   : spatial_batch_val,
            label_batch     : label_val
        })
    classification_loss_avg = decay*classification_loss_avg + (1-decay)*classification_loss_val
    print('\titer = %d, classification_loss (cur) = %f, classification_loss (avg) = %f, learningrate = %f'
        % (n_iter, classification_loss_val, classification_loss_avg, learningrate_val))

    # Accuracy
    accuracy_all, accuracy_positive, accuracy_negative = segmodel.compute_accuracy(scores_val, label_val)
    avg_accuracy_all = decay*avg_accuracy_all + (1-decay)*accuracy_all
    avg_accuracy_positive = decay*avg_accuracy_positive + (1-decay)*accuracy_positive
    avg_accuracy_negative = decay*avg_accuracy_negative + (1-decay)*accuracy_negative
    print('\titer = %d, accuracy (cur) = %f (all), %f (positive), %f (negative)'
          % (n_iter, accuracy_all, accuracy_positive, accuracy_negative))
    print('\titer = %d, accuracy (avg) = %f (all), %f (positive), %f (negative)'
          % (n_iter, avg_accuracy_all, avg_accuracy_positive, avg_accuracy_negative))

    # Save snapshot
    if (n_iter+1) % snapshot == 0 or (n_iter+1) == max_iter:
        snapshot_saver.save(sess, snapshot_file % (n_iter+1))
        print('snapshot saved to ' + snapshot_file % (n_iter+1))

print('Optimization done.')
sess.close()
