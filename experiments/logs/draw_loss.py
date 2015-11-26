#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re
import numpy as np
if sys.platform.startswith('linux'):
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math



def save_loss_curve(fname, num_subplot):
    loss_iter = []
    loss = []

    net_output_0 = []
    net_output_1 = []
    net_output_2 = []
    
    for line in open(fname):
        if 'Iteration' in line and 'loss' in line:
            txt = re.search(ur'Iteration\s([0-9]+)', line)
            loss_iter.append(int(txt.groups()[0]))
            txt = re.search(ur'loss\s=\s([0-9\.]+)\n', line)
            loss.append(float(txt.groups()[0]))
        if 'output' in line and '#0' in line:
            txt = re.search(ur'loss_bbox\s=\s([0-9\.]+)', line)
            net_output_0.append(float(txt.groups()[0]))
        if 'output' in line and '#1' in line:
            txt = re.search(ur'loss_cls\s=\s([0-9\.]+)', line)
            net_output_1.append(float(txt.groups()[0]))   
        if 'output' in line and '#2' in line:
            txt = re.search(ur'loss_pose\s=\s([0-9\.]+)', line)
            net_output_2.append(float(txt.groups()[0]))   
    
    print len(loss_iter), len(loss)
    if len(loss) < len(loss_iter):
        loss_iter = loss_iter[0:len(loss)]

    loss = [math.log(l) for l in loss]
    net_output_0 = [math.log(l) for l in net_output_0]
    net_output_1 = [math.log(l) for l in net_output_1]
    net_output_2 = [math.log(l) for l in net_output_2]
    
    lim = -9

    plt.clf()
    plt.ylim(lim, 0)
    num_subplot = int(num_subplot)
    f, axarr = plt.subplots(num_subplot, figsize=(6,9), sharex=True)
    for sub_id in range(num_subplot):
        if sub_id==0:
            axarr[sub_id].plot(loss_iter, loss, 'k')
            axarr[sub_id].set_title('loss')
        if sub_id==1:
            axarr[sub_id].plot(loss_iter, net_output_1, 'r')
            axarr[sub_id].set_title('loss_cls')
        if sub_id==2:
            axarr[sub_id].plot(loss_iter, net_output_0, 'g')
            axarr[sub_id].set_title('loss_bbox')
        if sub_id==3:
            axarr[sub_id].plot(loss_iter, net_output_2, 'b')
            axarr[sub_id].set_title('loss_pose')

    plt.savefig('loss_curve.png')
    plt.show()


if __name__ == '__main__':
    # argv[1]: log_file
    # argv[2]: num_subplot
    save_loss_curve(sys.argv[1], sys.argv[2])
