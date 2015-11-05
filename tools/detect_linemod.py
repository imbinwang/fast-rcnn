#!/usr/bin/env python
#
# author: Bin Wang
# date: 2015/10/18
#
# Use F-RCNN to detect objects in LINEMOD dataset.
#

"""
Demo script showing detections in test images.
"""

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import pprint

LINEMO_DATA_PATH = '/mnt/wb/dataset/LINEMOD4FRCNN'
DEEP_FITTING_ROOT_DIR = '/mnt/wb/project/DeepFitting'

CLASSES = ('__background__',
           'ape', 'benchviseblue', 'bowl', 'cam', 'can', 
           'cat', 'cup', 'driller', 'duck', 'eggbox', 
           'glue', 'holepuncher', 'iron', 'lamp', 'phone')

NETS = {'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_1200000.caffemodel')}

DATASETS = {'test': (LINEMO_DATA_PATH + '/data/ImageSets/test.txt',
                  LINEMO_DATA_PATH + '/test.mat')}

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def detect(net, im, obj_proposal, classes):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, obj_proposal)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    is_first_cls = True
    for cls in classes:
        cls_ind = CLASSES.index(cls)
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        keep = np.where(cls_scores >= CONF_THRESH)[0]
        cls_boxes = cls_boxes[keep, :]
        cls_scores = cls_scores[keep]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        print 'All {} detections with p({} | box) >= {:.1f}'.format(cls, cls,
                                                                    CONF_THRESH)
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='det_net', help='Network to use [caffenet]',
                        choices=NETS.keys(), default='caffenet')
    parser.add_argument('--cfg', dest='cfg_file',help='optional config file', 
                        default=None, type=str)
    parser.add_argument('--dataset', dest='data_set', help='Test data',
                        choices=DATASETS.keys(), default='test')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    prototxt = os.path.join(DEEP_FITTING_ROOT_DIR, 'src', 'fast_rcnn', NETS[args.det_net][0],
                            'test_linemod.prototxt')
    caffemodel = os.path.join(DEEP_FITTING_ROOT_DIR, 'output', 'LINEMOD',
                              NETS[args.det_net][1])

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    print('Using config:')
    pprint.pprint(cfg)

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/scripts/'
                       'fetch_fast_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    images = DATASETS[args.data_set][0]
    proposals = DATASETS[args.data_set][1]

    image_pathes = ['{}/{}/{}{}'.format(LINEMO_DATA_PATH, 'data/Images', line.strip(), '.jpg') for line in open(images).readlines()]
    det_image_pathes = ['{}/{}/{}{}'.format(DEEP_FITTING_ROOT_DIR, 'output/LINEMOD/caffenet_fast_rcnn_detection', line.strip(), '_det.jpg') for line in open(images).readlines()]
    images_inteval = [(0,411), (412, 816), (817,1227), (1228,1628), (1629,2027), 
                      (2028,2420), (2421,2834), (2835,3230), (3231,3648), (3649,4066), 
                      (4067,4473), (4474,4886), (4887,5270), (5271,5679), (5680,6094)]

    # Load pre-computed Selected Search object proposals
    obj_proposals = sio.loadmat(proposals)['boxes'].ravel()
    # detect on each image
    for i in range(len(image_pathes)):
        if i>=0 and i<=411:
            classes = (CLASSES[1],)
        if i>=412 and i<=816:
            classes = (CLASSES[2],)
        if i>=817 and i<=1227:
            classes = (CLASSES[3],)
        if i>=1228 and i<=1628:
            classes = (CLASSES[4],)  
        if i>=1629 and i<=2027:
            classes = (CLASSES[5],)  
        if i>=2028 and i<=2420:
            classes = (CLASSES[6],) 
        if i>=2421 and i<=2834:
            classes = (CLASSES[7],)  
        if i>=2835 and i<=3230:
            classes = (CLASSES[8],) 
        if i>=3231 and i<=3648:
            classes = (CLASSES[9],)
        if i>=3649 and i<=4066:
            classes = (CLASSES[10],)
        if i>=4067 and i<=4473:
            classes = (CLASSES[11],)
        if i>=4474 and i<=4886:
            classes = (CLASSES[12],)
        if i>=4887 and i<=5270:
            classes = (CLASSES[13],)
        if i>=5271 and i<=5679:
            classes = (CLASSES[14],)
        if i>=5680 and i<=6094:
            classes = (CLASSES[15],)
        im = cv2.imread(image_pathes[i])
        detect(net, im, obj_proposals[i], classes)
        plt.savefig(det_image_pathes[i])
