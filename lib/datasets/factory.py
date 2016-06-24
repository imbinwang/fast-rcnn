# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# this file is modified by Bin Wang(binwangsdu@gmail.com)
# use Fast R-CNN to detect LINEMOD dataset

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets.linemod
import datasets.linemod_sub
import datasets.pascal_voc
import numpy as np


### my own dataset ###

#------ real images for training and testing ------
linemod_devkit_path = '/mnt/wb/dataset/LINEMOD4FRCNN'
for split in ['train', 'test']:
    name = '{}_{}'.format('linemod', split)
    __sets[name] = (lambda split=split: datasets.linemod(split, linemod_devkit_path))
#------ real images ------

#------ synthesised images for training, real images for testing ------
linemod_comp_devkit_path = '/mnt/wb/dataset/LINEMOD_COMP4FRCNN'
for split in ['train', 'test']:
    name = '{}_{}'.format('linemod_comp', split)
    __sets[name] = (lambda split=split: datasets.linemod(split, linemod_comp_devkit_path))
#------ synthesised images for training, real images for testing ------

linemod_v_devkit_path = '/mnt/wb/dataset/LINEMOD_V4FRCNN'
for split in ['train', 'test']:
    name = '{}_{}'.format('linemod_v', split)
    __sets[name] = (lambda split=split: datasets.linemod(split, linemod_v_devkit_path))

linemod_bg_devkit_path = '/mnt/wb/dataset/LINEMOD_BG4FRCNN'
for split in ['train', 'test']:
    name = '{}_{}'.format('linemod_bg', split)
    __sets[name] = (lambda split=split: datasets.linemod(split, linemod_bg_devkit_path))

linemod_black_devkit_path = '/mnt/wb/dataset/LINEMOD_BLACK4FRCNN'
for split in ['train', 'test']:
    name = '{}_{}'.format('linemod_black', split)
    __sets[name] = (lambda split=split: datasets.linemod(split, linemod_black_devkit_path))

linemod_ror_devkit_path = '/mnt/wb/dataset/LINEMOD_ROR4FRCNN'
for split in ['train', 'test']:
    name = '{}_{}'.format('linemod_ror', split)
    __sets[name] = (lambda split=split: datasets.linemod_sub(split, linemod_ror_devkit_path))

linemod_bgsub_devkit_path = '/mnt/wb/dataset/LINEMOD_BGSUB4FRCNN'
for split in ['train', 'test']:
    name = '{}_{}'.format('linemod_bgsub', split)
    __sets[name] = (lambda split=split: datasets.linemod_sub(split, linemod_bgsub_devkit_path))

linemod_largesub_devkit_path = '/mnt/wb/dataset/LINEMOD_LARGE4FRCNN'
for split in ['train', 'test']:
    name = '{}_{}'.format('linemod_largesub', split)
    __sets[name] = (lambda split=split: datasets.linemod_sub(split, linemod_largesub_devkit_path))

### my own dataset ###

def _selective_search_IJCV_top_k(split, year, top_k):
    """Return an imdb that uses the top k proposals from the selective search
    IJCV code.
    """
    imdb = datasets.pascal_voc(split, year)
    imdb.roidb_handler = imdb.selective_search_IJCV_roidb
    imdb.config['top_k'] = top_k
    return imdb

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year:
                datasets.pascal_voc(split, year))

# Set up voc_<year>_<split>_top_<k> using selective search "quality" mode
# but only returning the first k boxes
for top_k in np.arange(1000, 11000, 1000):
    for year in ['2007', '2012']:
        for split in ['train', 'val', 'trainval', 'test']:
            name = 'voc_{}_{}_top_{:d}'.format(year, split, top_k)
            __sets[name] = (lambda split=split, year=year, top_k=top_k:
                    _selective_search_IJCV_top_k(split, year, top_k))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
