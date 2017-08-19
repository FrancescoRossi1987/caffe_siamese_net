#coding:utf-8
#########################################################################
# File Name: train.py
# Author:Lei Jiang
# mail: jianglei@1000look.com
# Created Time: 2016年03月28日 星期一 09时21分43秒
# Copyright Nanjing Qing So information technology
#########################################################################

from params import Params
pa=Params()
caffe_root=pa.caffe_root
import sys
sys.path.insert(0,caffe_root)
import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2
import os.path as osp
import numpy as np


class SolverWrapper(object):
    def __init__(self):
        if pa.GPU==True:
            caffe.set_device(pa.device)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.solver=caffe.SGDSolver(pa.solver)
        if pa.pretrain!="":
            self.solver.net.copy_from(pa.pretrain)
        self.solver_param=caffe_pb2.SolverParameter()
        with open(pa.solver,'rt') as f:
            pb2.text_format.Merge(f.read(),self.solver_param)

        #self.output_dir=pa.output_dir
        self.solver.net.layers[0].set_queue()
    
    def snap_shot(self):
        net=self.solver.net
        filename=(self.solver_param.snapshot_prefix+'_iter_{:d}'.format(self.solver.iter)+'.caffemodel')
        #filename=osp.join(self.output_dir,filename)
        net.save(str(filename))
        print "Wrote snapshot to {:s}".format(filename)

    def train_mode(self):
        '''Network train looping...'''
        print "------------training start---------------"
        last_snapshot_iter=-1
        while self.solver.iter<self.solver_param.max_iter:
            self.solver.step(1) #update once
            if self.solver.iter % self.solver_param.snapshot ==0:
                last_snapshot_iter=self.solver.iter
                self.snap_shot()
            if self.solver.iter % self.solver_param.test_interval ==0:
                print "------------Iteration testing----------------"
                
        if last_snapshot_iter!=self.solver.iter:
            self.snap_shot()



        




        

