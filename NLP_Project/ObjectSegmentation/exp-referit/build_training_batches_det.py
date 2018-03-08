from __future__ import absolute_import, division, print_function

import numpy as np
import os
import json
import skimage
import skimage.io
import skimage.transform

from util import image_processing, text_processing, eval_tools
from models import processing_tools

# Parameters
image_loc = './exp-referit/referit-dataset/images/'
bbox_proposal_loc = './exp-referit/data/referit_edgeboxes_top100/'
expression_file = './exp-referit/data/referit_query_trainval.json'
bbox_file = './exp-referit/data/referit_bbox.json'
imagecrop_file = './exp-referit/data/referit_imcrop.json'
imagesize_file = './exp-referit/data/referit_imsize.json'
vocabulary_file = './exp-referit/data/vocabulary_referit.txt'

# Saving directory
data_folder = './exp-referit/data/train_batch_det/'
data_prefix = 'referit_train_det'

# Sample selection params
positive_iou = .7
negative_iou = 1e-6
negative_to_positive_ratio = 1.0

# Model Param
N = 50
T = 20


query_dictionary = json.load(open(expression_file))
bbox_dict = json.load(open(bbox_file))
imagecrop_dict = json.load(open(imagecrop_file))
imagesize_dict = json.load(open(imagesize_file))
imagelist = list({name.split('_', 1)[0] + '.jpg' for name in query_dictionary})
vocabulary_dict = text_processing.load_vocab_dict_from_file(vocabulary_file)

# Object proposals
bbox_proposal_dictionary = {}
for imname in imagelist:
    bboxes = np.loadtxt(bbox_proposal_loc + imname[:-4] + '.txt').astype(int).reshape((-1, 4))
    bbox_proposal_dictionary[imname] = bboxes

# Load training data
training_samples_positive = []
training_samples_negative = []
for imagename in imagelist:
    this_imagecrop_names = imagecrop_dict[imagename]
    imsize = imagesize_dict[imagename]
    bbox_proposals = bbox_proposal_dictionary[imagename]
    # for each ground-truth annotation, use gt box and proposal boxes as positive examples
    # and proposal box with small iou as negative examples
    for imagecrop_name in this_imagecrop_names:
        if not imagecrop_name in query_dictionary:
            continue
        gt_bbox = np.array(bbox_dict[imagecrop_name]).reshape((1, 4))
        IoUs = eval_tools.compute_bbox_iou(bbox_proposals, gt_bbox)
        pos_boxes = bbox_proposals[IoUs >= positive_iou, :]
        pos_boxes = np.concatenate((gt_bbox, pos_boxes), axis=0)
        neg_boxes = bbox_proposals[IoUs <  negative_iou, :]

        this_descriptions = query_dictionary[imagecrop_name]
        # generate them per discription
        for description in this_descriptions:
            # Positive training samples
            for n_pos in range(pos_boxes.shape[0]):
                sample = (imagename, imsize, pos_boxes[n_pos], description, 1)
                training_samples_positive.append(sample)
            # Negative training samples
            for n_neg in range(neg_boxes.shape[0]):
                sample = (imagename, imsize, neg_boxes[n_neg], description, 0)
                training_samples_negative.append(sample)

# Print numbers of positive and negative samples
print('#pos=', len(training_samples_positive))
print('#neg=', len(training_samples_negative))

# Subsample negative training data
np.random.seed(3)
sample_idx = np.random.choice(len(training_samples_negative),
                              min(len(training_samples_negative),
                                  int(negative_to_positive_ratio*len(training_samples_positive))),
                              replace=False)
training_samples_negative_subsample = [training_samples_negative[n] for n in sample_idx]
print('#neg_subsample=', len(training_samples_negative_subsample))

# Merge and shuffle training examples
training_samples = training_samples_positive + training_samples_negative_subsample
np.random.seed(3)
perm_idx = np.random.permutation(len(training_samples))
shuffled_training_samples = [training_samples[n] for n in perm_idx]
del training_samples
print('#total sample=', len(shuffled_training_samples))

num_batch = len(shuffled_training_samples) // N
print('total batch number: %d' % num_batch)

# Save training samples to disk
text_seq_batch = np.zeros((T, N), dtype=np.int32)
imagecrop_batch = np.zeros((N, 224, 224, 3), dtype=np.uint8)
spatial_batch = np.zeros((N, 8), dtype=np.float32)
label_batch = np.zeros((N, 1), dtype=np.bool)

if not os.path.isdir(data_folder):
    os.mkdir(data_folder)
for n_batch in range(num_batch):
    print('saving batch %d / %d' % (n_batch+1, num_batch))
    batch_begin = n_batch * N
    batch_end = (n_batch+1) * N
    for n_sample in range(batch_begin, batch_end):
        imagename, imsize, sample_bbox, description, label = shuffled_training_samples[n_sample]
        im = skimage.io.imread(image_loc + imagename)
        xmin, ymin, xmax, ymax = sample_bbox

        imagecrop = im[ymin:ymax+1, xmin:xmax+1, :]
        imagecrop = skimage.img_as_ubyte(skimage.transform.resize(imagecrop, [224, 224]))
        spatial_feat = processing_tools.spatial_feature_from_bbox(sample_bbox, imsize)
        text_seq = text_processing.preprocess_sentence(description, vocabulary_dict, T)

        idx = n_sample - batch_begin
        text_seq_batch[:, idx] = text_seq
        imagecrop_batch[idx, ...] = imagecrop
        spatial_batch[idx, ...] = spatial_feat
        label_batch[idx] = label

    np.savez(file=data_folder + data_prefix + '_' + str(n_batch) + '.npz',
        text_seq_batch=text_seq_batch,
        imagecrop_batch=imagecrop_batch,
        spatial_batch=spatial_batch,
        label_batch=label_batch)
