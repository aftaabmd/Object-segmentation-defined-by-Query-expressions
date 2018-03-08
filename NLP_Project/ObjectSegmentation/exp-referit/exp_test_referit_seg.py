from __future__ import absolute_import, division, print_function

import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
import skimage.io
import numpy as np
import tensorflow as tf
import json
import timeit

from models import text_objseg_model as segmodel
from util import image_processing, text_processing, eval_tools
from util.io import load_referit_gt_mask as load_gt_mask

# Parameters
image_dir = './exp-referit/referit-dataset/images/'
mask_dir = './exp-referit/referit-dataset/mask/'
query_file = './exp-referit/data/referit_query_test.json'
bbox_file = './exp-referit/data/referit_bbox.json'
imagecrop_file = './exp-referit/data/referit_imcrop.json'
imagesize_file = './exp-referit/data/referit_imsize.json'
vocab_file = './exp-referit/data/vocabulary_referit.txt'

pretrained_model = './exp-referit/tfmodel/referit_fc8_segment_highresolution_iteration_18000.tfmodel'

# Model Param
T = 20
N = 1
#height
input_H = 512; featmap_H = (input_H // 32)
#weight
input_W = 512; featmap_W = (input_W // 32)
vocab_size = 8803
embedded_dim = 1000
lstm_dim = 1000
mlp_hidden_dims = 500

# Evaluation Param
score_threshold = 1e-9

# Evaluation network

# Inputs
text_seq_batch = tf.placeholder(tf.int32, [T, N])
imagecrop_batch = tf.placeholder(tf.float32, [N, input_H, input_W, 3])

# Outputs
scores = segmodel.text_objseg_upsample32s(text_seq_batch, imagecrop_batch,
    vocab_size, embedded_dim, lstm_dim, mlp_hidden_dims, vgg_dropout=False,
    mlp_dropout=False)

# Load pretrained model
snapshot_restorer = tf.train.Saver()
sess = tf.Session()
snapshot_restorer.restore(sess, pretrained_model)

query_dict = json.load(open(query_file))
bbox_dict = json.load(open(bbox_file))
imagecrop_dict = json.load(open(imagecrop_file))
imagesize_dict = json.load(open(imagesize_file))
imlist = list({name.split('_', 1)[0] + '.jpg' for name in query_dict})
vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

# Flatten the annotations
flat_query_dict = {imname: [] for imname in imlist}
for imname in imlist:
    this_imagecrop_names = imagecrop_dict[imname]
    for imagecrop_name in this_imagecrop_names:
        gt_bbox = bbox_dict[imagecrop_name]
        if imagecrop_name not in query_dict:
            continue
        this_descriptions = query_dict[imagecrop_name]
        for description in this_descriptions:
            flat_query_dict[imname].append((imagecrop_name, gt_bbox, description))

# Testing
cum_I, cum_U = 0, 0
eval_seg_iou_list = [.5, .6, .7, .8, .9]
correct_seg = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
seg_total = 0

text_seq_val = np.zeros((T, N), dtype=np.float32)
imagecrop_val = np.zeros((N, input_H, input_W, 3), dtype=np.float32)

num_im = len(imlist)
for n_im in range(num_im):
    print('testing image %d / %d' % (n_im, num_im))
    imname = imlist[n_im]

    # Extract visual features from all proposals
    im = skimage.io.imread(image_dir + imname)
    processed_im = skimage.img_as_ubyte(image_processing.resize_and_pad(im, input_H, input_W))
    if processed_im.ndim == 2:
        processed_im = np.tile(processed_im[:, :, np.newaxis], (1, 1, 3))

    imagecrop_val[...] = processed_im.astype(np.float32) - segmodel.vgg_net.channel_mean
    for imagecrop_name, _, description in flat_query_dict[imname]:
        mask = load_gt_mask(mask_dir + imagecrop_name + '.mat').astype(np.float32)
        labels = (mask > 0)
        processed_labels = image_processing.resize_and_pad(mask, input_H, input_W) > 0

        text_seq_val[:, 0] = text_processing.preprocess_sentence(description, vocab_dict, T)
        scores_val = sess.run(scores, feed_dict={
                text_seq_batch  : text_seq_val,
                imagecrop_batch    : imagecrop_val
            })
        scores_val = np.squeeze(scores_val)

        # Evaluate the segmentation performance of using bounding box segmentation
        pred_raw = (scores_val >= score_threshold).astype(np.float32)
        predicts = image_processing.resize_and_crop(pred_raw, im.shape[0], im.shape[1])
        I, U = eval_tools.compute_mask_IU(predicts, labels)
        cum_I += I
        cum_U += U
        this_IoU = I/U
        for n_eval_iou in range(len(eval_seg_iou_list)):
            eval_seg_iou = eval_seg_iou_list[n_eval_iou]
            correct_seg[n_eval_iou] += (I/U >= eval_seg_iou)
        seg_total += 1

# Print results
print('Final results on the whole test set')
result_str = ''
for n_eval_iou in range(len(eval_seg_iou_list)):
    result_str += 'precision@%s = %f\n' % \
        (str(eval_seg_iou_list[n_eval_iou]), correct_seg[n_eval_iou]/seg_total)
result_str += 'overall IoU = %f\n' % (cum_I/cum_U)
print(result_str)
