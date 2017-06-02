# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

# Original license text is below
# BSD 2-Clause License
#
# Copyright (c) 2016, Bichen Wu
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""SqueezeDet Demo.

In image detection mode, for a given image, detect objects and draw bounding
boxes around them. In video detection mode, perform real-time detection on the
video stream.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
import sys
import os
import glob

import numpy as np
import tensorflow as tf

from config import *
from train import _draw_box
from nets import *
from utils.util import bbox_transform

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'checkpoint', './data/model_checkpoints/squeezeDet/model.ckpt-87000',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'out_dir', './data/out/', """Directory to dump output image or video.""")
tf.app.flags.DEFINE_string(
    'image_dir', './', """Directory with images""")
tf.app.flags.DEFINE_string(
    'label_dir', './', """Directory with image labels""")
tf.app.flags.DEFINE_float(
    'iou_threshold', 0.7, """Threshold for IoU metric to determine false positives""")
tf.app.flags.DEFINE_string(
    'demo_net', 'squeezeDet', """Neural net architecture.""")

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def my_draw_box(im, box_list, label_list, color=(0,255,0), cdict=None, label_placement='bottom'):
    assert label_placement == 'bottom' or label_placement == 'top', \
        'label_placement format not accepted: {}.'.format(label_placement)

    for bbox, label in zip(box_list, label_list):
        xmin, ymin, xmax, ymax = [int(b) for b in bbox]

        l = label.split(':')[0] # text before "CLASS: (PROB)"
        if cdict and l in cdict:
            c = cdict[l]
        else:
            c = color

        # draw box
        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), c, 1)
        # draw label
        font = cv2.FONT_HERSHEY_SIMPLEX
        if label_placement == 'bottom':
            cv2.putText(im, label, (xmin, ymax), font, 0.3, c, 1)
        else:
            cv2.putText(im, label, (xmin, ymin), font, 0.3, c, 1)

class Box:
    def __init__(self, klass, bbox, occlusion=0, truncation=0, alpha=0, prob=0):
        self.klass = klass
        self.bbox = bbox
        self.occlusion = occlusion
        self.truncation = truncation
        self.alpha = alpha
        self.prob = prob
        self.assigned = False

    def __str__(self):
        b = self
        bb = b.bbox
        return '{} {:.2f} {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(b.klass, b.truncation, b.occlusion, b.alpha, bb[0], bb[1], bb[2], bb[3], b.prob)

def print_boxes(prefix, boxes):
    for b in boxes:
        print('{} {}'.format(prefix, b))

def image_demo():
  """Detect image."""

  assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+' or FLAGS.demo_net == 'resnet50' or FLAGS.demo_net == 'vgg16', \
    'Selected nueral net architecture not supported: {}'.format(FLAGS.demo_net)

  with tf.Graph().as_default():
    # Load model
    if FLAGS.demo_net == 'squeezeDet':
      mc = kitti_squeezeDet_config()
      mc.BATCH_SIZE = 1
      # model parameters will be restored from checkpoint
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDet(mc, FLAGS.gpu)
    elif FLAGS.demo_net == 'squeezeDet+':
      mc = kitti_squeezeDetPlus_config()
      mc.BATCH_SIZE = 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDetPlus(mc, FLAGS.gpu)
    elif FLAGS.demo_net == 'resnet50':
      mc = kitti_res50_config()
      mc.BATCH_SIZE = 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = ResNet50ConvDet(mc, FLAGS.gpu)
    elif FLAGS.demo_net == 'vgg16':
      mc = kitti_vgg16_config()
      mc.BATCH_SIZE = 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = VGG16ConvDet(mc, FLAGS.gpu)

    saver = tf.train.Saver(model.model_params)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver.restore(sess, FLAGS.checkpoint)

      class_names = [k.lower() for k in mc.CLASS_NAMES]
      d = FLAGS.image_dir
      image_list = sorted([os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))])

      for f in image_list:
        im = cv2.imread(f)
        im = im.astype(np.float32, copy=False)
        im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
        input_image = im - mc.BGR_MEANS

        start_clock = time.clock()

        # Detect
        det_boxes, det_probs, det_class = sess.run(
            [model.det_boxes, model.det_probs, model.det_class],
            feed_dict={model.image_input:[input_image]})

        # Filter
        final_boxes, final_probs, final_class = model.filter_prediction(
            det_boxes[0], det_probs[0], det_class[0])

        duration = time.clock() - start_clock

        keep_idx    = [idx for idx in range(len(final_probs)) \
                          if final_probs[idx] > mc.PLOT_PROB_THRESH]
        final_boxes = [final_boxes[idx] for idx in keep_idx]
        final_probs = [final_probs[idx] for idx in keep_idx]
        final_class = [final_class[idx] for idx in keep_idx]

        recognized = [Box(class_names[k], bbox_transform(bbox), prob=p) for k, bbox, p in zip(final_class, final_boxes, final_probs)]

        # TODO(bichen): move this color dict to configuration file
        cls2clr = {
            'car': (255, 191, 0),
            'cyclist': (0, 191, 255),
            'pedestrian':(255, 0, 191)
        }

        file_name = os.path.split(f)[1]

        expected = []
        dontcare = []

        class_count = dict((k, 0) for k in mc.CLASS_NAMES)
        
        if FLAGS.label_dir:
            label_file_name = os.path.join(FLAGS.label_dir, file_name)
            label_file_name = os.path.splitext(label_file_name)[0] + '.txt'
            with open(label_file_name) as lf:
                label_lines = [x.strip() for x in lf.readlines()]
                for l in label_lines:
                    parts = l.strip().lower().split(' ')
                    klass = parts[0]
                    bbox = [float(parts[i]) for i in [4, 5, 6, 7]]
                    if klass in class_count.keys():
                        class_count[klass] += 1
                        b = Box(klass, bbox, truncation=float(parts[1]), occlusion=float(parts[2]), alpha=float(parts[3]), prob=1)
                        expected.append(b)
                    elif klass == 'dontcare':
                        dontcare.append(Box(klass, bbox))

        expected_class_count = class_count

        # Draw original boxes
        my_draw_box(
            im, [b.bbox for b in expected],
            [b.klass + ': (TRUE)' for b in expected],
            label_placement='top', color=(200,200,200)
        )

        # Draw recognized boxes
        my_draw_box(
            im, [b.bbox for b in recognized],
            [b.klass + ': (%.2f)' % b.prob for b in recognized],
            cdict=cls2clr
        )

        out_file_name = os.path.join(FLAGS.out_dir, 'out_'+file_name)
        cv2.imwrite(out_file_name, im)
        
        print('File: {}'.format(out_file_name))
        print('Duration: {} sec'.format(duration))

        class_count = dict((k.lower(), 0) for k in mc.CLASS_NAMES)
        for k in final_class:
            class_count[class_names[k]] += 1

        for k, v in class_count.items():
            print('Recognized {}: {}'.format(k, v))

        for k, v in expected_class_count.items():
            print('Expected {}: {}'.format(k, v))

        false_positives_count = dict((k, 0) for k in class_names)
        threshold = FLAGS.iou_threshold
        for recb in recognized:
            for gtb in [b for b in expected if not b.assigned]:
                iou = bb_intersection_over_union(recb.bbox, gtb.bbox)
                if iou >= threshold:
                    gtb.assigned = True
                    recb.assigned = True
                    break

            if not recb.assigned:
                false_positives_count[recb.klass] += 1

        for k, v in false_positives_count.items():
            print('False positive {}: {}'.format(k, v))

        print_boxes('Ground truth', expected + dontcare)
        print_boxes('Detection', recognized)

        print('')
        sys.stdout.flush()

def main(argv=None):
  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)
  image_demo()

if __name__ == '__main__':
    tf.app.run()
