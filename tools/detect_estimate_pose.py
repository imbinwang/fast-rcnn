#!/usr/bin/env python
#
# author: Bin Wang
# date: 2015/11/16
#
# Use F-RCNN to detect objects in LINEMOD dataset.
# Simultaneous estimate pose.
#

"""
Demo script showing detections in test images.
"""

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect, im_detect_pose
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import pprint
import shutil

DATA_PATH = '/mnt/wb/dataset/LINEMOD4FRCNN'
FRCNN_ROOT_DIR = '/mnt/wb/project/fast-rcnn'

CLASSES = ('__background__',
           'ape', 'benchviseblue', 'bowl', 'cam', 'can', 
           'cat', 'cup', 'driller', 'duck', 'eggbox', 
           'glue', 'holepuncher', 'iron', 'lamp', 'phone')

NETS = {'detect_net_1200k': ('wb_CaffeNet',
                     'test_linemod.prototxt',
                     'caffenet_fast_rcnn_2_iter_1200000.caffemodel'),
        'detect_net_30k': ('wb_CaffeNet',
                     'test_linemod.prototxt',
                     'caffenet_fast_rcnn_iter_30000.caffemodel'),
        'pose_net_1200k': ('wb_CaffeNet',
                     'test_linemod_pose_reg.prototxt',
                     'caffenet_fast_rcnn_pose_reg_2_iter_1200000.caffemodel'),
        'pose_net_30k': ('wb_CaffeNet',
                     'test_linemod_pose_reg.prototxt',
                     'caffenet_fast_rcnn_pose_reg_iter_30000.caffemodel')}

DATASETS = {'linemod_test': (DATA_PATH + '/data/ImageSets/test.txt',
                  DATA_PATH + '/test.mat')}

def vis_detections(im, classes_name, all_dets, pose_reg, img_path=None, txt_path=None, thresh=0.5):
    """Draw detected bounding boxes."""

    if txt_path is not None:
        txt_file = open(txt_path, 'w')
    im = im[:, :, (2, 1, 0)]
    im_h = im.shape[0]
    im_w = im.shape[1]
    DPI = 100.0
    fig, ax = plt.subplots(figsize=(im_w/DPI, im_h/DPI))
    ax.imshow(im, aspect='equal')
    for cls in xrange(len(all_dets)):
        dets = all_dets[cls]
        inds = np.where(dets[:, 4] >= thresh)[0]
        if len(inds) == 0:
            continue
        for i in inds:
            bbox = dets[i, 0:4]
            score = dets[i, 4]
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=2.0)
                )
            if txt_path is not None:
                txt_file.write('{0} {1} {2[0]} {2[1]} {2[2]} {2[3]}'.format(classes_name[cls], score, bbox))
            if pose_reg:
                pose = dets[i,5:9] 
                ax.text(bbox[0], bbox[1] - 2,
                        '{:s} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(classes_name[cls], score,
                         pose[0],  pose[1],  pose[2],  pose[3]),
                        bbox=dict(facecolor='blue', alpha=0.5),
                        fontsize=10, color='white')
                if txt_path is not None:
                    txt_file.write(' {0[0]} {0[1]} {0[2]} {0[3]}\n'.format(pose))
            else:
                ax.text(bbox[0], bbox[1] - 2,
                        '{:s} {:.3f}'.format(classes_name[cls], score),
                        bbox=dict(facecolor='blue', alpha=0.5),
                        fontsize=10, color='white')
                if txt_path is not None:
                    txt_file.write('\n')
    plt.axis('off')
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
    #plt.tight_layout()
    plt.draw()
   
    if img_path is not None:
        plt.savefig(img_path, dpi=100)
    if txt_path is not None:
        txt_file.close()

def detect_estimate_pose(net, im, obj_proposal, classes, pose_reg, img_path, txt_path):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    if pose_reg:
        scores, boxes, poses = im_detect_pose(net, im, obj_proposal)
    else:
        scores, boxes = im_detect(net, im, obj_proposal)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    all_dets = [[] for _ in xrange(len(classes))]
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    i = 0
    for cls in classes:
        cls_ind = CLASSES.index(cls)
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        keep = np.where(cls_scores >= CONF_THRESH)[0]
        cls_boxes = cls_boxes[keep, :]
        cls_scores = cls_scores[keep]

        if pose_reg:
            cls_poses = poses[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_poses = cls_poses[keep, :]        

        all_dets[i] = np.hstack((cls_boxes, 
                                 cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(all_dets[i], NMS_THRESH)
        all_dets[i] = all_dets[i][keep, :]

        if pose_reg:
            cls_poses = cls_poses[keep, :]
            all_dets[i] = np.hstack((all_dets[i], 
                                cls_poses)).astype(np.float32)

        i += 1
        print 'All {} detections with p({} | box) >= {:.1f}'.format(cls, cls,
                                                                    CONF_THRESH)

    vis_detections(im, classes, all_dets, pose_reg, img_path, txt_path, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='net', help='Network to use [caffenet]',
                        choices=NETS.keys(), default='caffenet')
    parser.add_argument('--cfg', dest='cfg_file',help='optional config file', 
                        default=None, type=str)
    parser.add_argument('--dataset', dest='data_set', help='Test data',
                        choices=DATASETS.keys(), default='linemod_test')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    prototxt = os.path.join(FRCNN_ROOT_DIR, 'models', NETS[args.net][0],
                            NETS[args.net][1])
    caffemodel = os.path.join(FRCNN_ROOT_DIR, 'output', 'LINEMOD',
                              NETS[args.net][2])

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

    image_pathes = ['{0}/{1}/{2}{3}'.format(DATA_PATH, 'data/Images', line.strip(), '.jpg') for line in open(images).readlines()]

    out_path = os.path.join(DATA_PATH, 'results', args.net)
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)
    det_image_pathes = ['{0}/{1}{2}'.format(out_path, line.strip(), '_det.jpg') for line in open(images).readlines()]
    det_txt_pathes = ['{0}/{1}{2}'.format(out_path, line.strip(), '_det.txt') for line in open(images).readlines()]

    # Load pre-computed Selected Search object proposals
    raw_boxes = sio.loadmat(proposals)['boxes'].ravel()
    # detect on each image
    for i in range(len(image_pathes)):
        im = cv2.imread(image_pathes[i])
        # change the box format from matlab(1-based index, colomn major) to python format(0-based index, row major)
        obj_proposals = raw_boxes[i][:, (1, 0, 3, 2)]-1
        detect_estimate_pose(net, im, obj_proposals, CLASSES[1:], cfg.TEST.POSE_REG, det_image_pathes[i], det_txt_pathes[i])
