#coding:utf-8
#########################################################################
# File Name: params.py
# Author:
# mail: 
# Created Time: 2016年03月23日 星期三 15时46分22秒
# Copyright Nanjing Qing So information technology
#########################################################################

import numpy as np

class Params(object):
    def __init__(self):
        self.caffe_root='../new_caffe/caffe/python'
        self.batchsize_dic={'train':32,'val':32}
        self.channel=3
        self.height=227
        self.width=227
        self.GPU=True
        self.data_label_dic={'train':['',''],
                             'val':['','']}
        self.top_names=['data','data_p','sim']
        self.mean=np.float32([128.0,128.0,128.0])
        self.solver='models/solver.pt'
        self.pretrain='/opt/jl/caffe-master-new/models/bvlc_caffenet/bvlc_caffenet.caffemodel'
        self.device=1
        self.output_dir='output'
        self.queue_num=10
