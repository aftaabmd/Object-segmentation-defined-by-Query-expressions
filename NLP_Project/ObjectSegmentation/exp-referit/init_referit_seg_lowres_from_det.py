from __future__ import absolute_import, division, print_function

import tensorflow as tf

from models import text_objseg_model as segmodel

# Parameters

det_model = './exp-referit/tfmodel/referit_fc8_detect_iteration_25000.tfmodel'
seg_model = './exp-referit/tfmodel/referit_fc8_segment_lowresolution_initialization.tfmodel'

# Model Params
T = 20
N = 1

vocab_size = 8803
embedded_dim = 1000
lstm_dim = 1000
mlp_hidden_dims = 500

# detection network
# Inputs
text_seq_batch = tf.placeholder(tf.int32, [T, N])  # one batch per sentence
imagecrop_batch = tf.placeholder(tf.float32, [N, 224, 224, 3])
spatial_batch = tf.placeholder(tf.float32, [N, 8])

# Language feature (LSTM hidden state)
_ = segmodel.text_objseg_region(text_seq_batch, imagecrop_batch,
    spatial_batch, vocab_size, embedded_dim, lstm_dim, mlp_hidden_dims,
    vgg_dropout=False, mlp_dropout=False)

# Load pretrained detection model and fetch weights
snapshot_loader = tf.train.Saver()
with tf.Session() as sess:
    snapshot_loader.restore(sess, det_model)
    variable_dict = {variable.name:variable.eval(session=sess) for variable in tf.all_variables()}

# low resolution segmentation network
# Clear the graph
tf.python.ops.reset_default_graph()

# Inputs
text_seq_batch = tf.placeholder(tf.int32, [T, N])  # one batch per sentence
imagecrop_batch = tf.placeholder(tf.float32, [N, 512, 512, 3])

_ = segmodel.text_objseg_full_conv(text_seq_batch, imagecrop_batch,
    vocab_size, embedded_dim, lstm_dim, mlp_hidden_dims,
    vgg_dropout=False, mlp_dropout=False)

# Assign outputs
assign_ops = []
for variable in tf.all_variables():
    assign_ops.append(tf.assign(variable, variable_dict[variable.name].reshape(variable.get_shape().as_list())))

# Save segmentation model initialization
snapshot_saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.group(*assign_ops))
    snapshot_saver.save(sess, seg_model)
