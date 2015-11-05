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



def save_loss_curve(fname):
    loss_iter = []
    loss = []

    net_output_0 = []
    net_output_1 = []
    
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
    
    print len(loss_iter), len(loss)
    if len(loss) < len(loss_iter):
        loss_iter = loss_iter[0:len(loss)]

    loss = [math.log(l) for l in loss]
    net_output_0 = [math.log(l) for l in net_output_0]
    net_output_1 = [math.log(l) for l in net_output_1]
    
    lim = -9

    plt.clf()
    plt.ylim(lim, 0)
    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(loss_iter, loss, 'b')
    axarr[1].plot(loss_iter, net_output_0, 'g')
    axarr[2].plot(loss_iter, net_output_0, 'r')
    plt.savefig('loss_curve.png')
    plt.show()


if __name__ == '__main__':
    save_loss_curve(sys.argv[1])
