from __future__ import absolute_import, division, print_function

import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
import tensorflow as tf
import numpy as np

from models import text_objseg_model as segmodel
from util import data_reader
from util import loss

# Model Params
T = 20
N = 10
input_H = 512; featmap_H = (input_H // 32)
input_W = 512; featmap_W = (input_W // 32)
vocab_size = 8803
embedded_dim = 1000
lstm_dim = 1000
mlp_hidden_dims = 500

# Initialization Params
pretrained_model = './exp-referit/tfmodel/referit_fc8_segment_lowresolution_initialization.tfmodel'

# Training Params
pos_loss_mult = 1.
neg_loss_mult = 1.

start_learningrate = 0.01
learningrate_decay_step = 10000
learningrate_decay_rate = 0.1
weight_decay = 0.005
momentum = 0.8
max_iter = 20000

fix_convnet = False
vgg_dropout = False
mlp_dropout = False
vgg_learningrate_mult = 1.

# Data Params
data_folder = './exp-referit/data/train_batch_seg/'
data_prefix = 'referit_train_seg'

# Snapshot Params
snapshot = 5000
snapshot_file = './exp-referit/tfmodel/referit_fc8_segment_lowresolution_initialization_%d.tfmodel'

# The model
# Inputs
text_seq_batch = tf.placeholder(tf.int32, [T, N])
imagecrop_batch = tf.placeholder(tf.float32, [N, input_H, input_W, 3])
label_batch = tf.placeholder(tf.float32, [N, featmap_H, featmap_W, 1])

# Outputs
scores = segmodel.text_objseg_full_conv(text_seq_batch, imagecrop_batch,
    vocab_size, embedded_dim, lstm_dim, mlp_hidden_dims,
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
reg_variable_list = [variable for variable in tf.trainable_variables()
                if (variable in train_variable_list) and
                (variable.name[-9:-2] == 'weights' or variable.name[-8:-2] == 'Matrix')]
print('Collecting variables for regularization:')
for variable in reg_variable_list: print('\t%s' % variable.name)
print('Done.')

# Collect learning rate for trainable variables
variable_learningrate_mult = {variable: (vgg_learningrate_mult if variable.name.startswith('vgg_local') else 1.0)
               for variable in train_variable_list}
print('Variable learning rate multiplication:')
for variable in train_variable_list:
    print('\t%s: %f' % (variable.name, variable_learningrate_mult[variable]))
print('Done.')

# Loss function and accuracy
cls_loss = loss.weighed_logistic_loss(scores, label_batch, pos_loss_mult, neg_loss_mult)
reg_loss = loss.l2_regularization_loss(reg_variable_list, weight_decay)
total_loss = cls_loss + reg_loss

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
snapshot_loader = tf.train.Saver(tf.trainable_variables())

# Load data
reader = data_reader.DataReader(data_folder, data_prefix)

snapshot_saver = tf.train.Saver()
sess = tf.Session()

# Run Initialization operations
sess.run(tf.initialize_all_variables())
snapshot_loader.restore(sess, pretrained_model)

# Optimization loop
cls_loss_avg = 0
avg_accuracy_all, avg_accuracy_positive, avg_accuracy_negative = 0, 0, 0
decay = 0.99

for n_iter in range(max_iter):
    # Read one batch
    batch = reader.read_batch()
    text_seq_val = batch['text_seq_batch']
    imagecrop_val = batch['imagecrop_batch'].astype(np.float32) - segmodel.vgg_net.channel_mean
    label_val = batch['label_coarse_batch'].astype(np.float32)

    # Forward and Backward pass
    scores_val, cls_loss_val, _, learningrate_val = sess.run([scores, cls_loss, train_step, learning_rate],
        feed_dict={
            text_seq_batch  : text_seq_val,
            imagecrop_batch    : imagecrop_val,
            label_batch     : label_val
        })
    cls_loss_avg = decay*cls_loss_avg + (1-decay)*cls_loss_val
    print('\titer = %d, cls_loss (cur) = %f, cls_loss (avg) = %f, learningrate = %f'
        % (n_iter, cls_loss_val, cls_loss_avg, learningrate_val))

    # Accuracy
    accuracy_all, accuracy_positive, accuracy_negative = segmodel.compute_accuracy(scores_val, label_val)
    avg_accuracy_all = decay*avg_accuracy_all + (1-decay)*accuracy_all
    avg_accuracy_positive = decay*avg_accuracy_positive + (1-decay)*accuracy_positive
    avg_accuracy_negative = decay*avg_accuracy_negative + (1-decay)*accuracy_negative
    print('\titer = %d, accuracy (cur) = %f (all), %f (pos), %f (neg)'
          % (n_iter, accuracy_all, accuracy_positive, accuracy_negative))
    print('\titer = %d, accuracy (avg) = %f (all), %f (pos), %f (neg)'
          % (n_iter, avg_accuracy_all, avg_accuracy_positive, avg_accuracy_negative))

    # Save snapshot
    if (n_iter+1) % snapshot == 0 or (n_iter+1) >= max_iter:
        snapshot_saver.save(sess, snapshot_file % (n_iter+1))
        print('snapshot saved to ' + snapshot_file % (n_iter+1))

print('Optimization done.')
sess.close()
