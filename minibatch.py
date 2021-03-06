#coding:utf-8
#########################################################################
# File Name: minibatch.py
# Author:Lei Jiang
# mail: leijiang@1000look.com
# Created Time: 2016年05月17日 星期二 16时48分08秒
# Copyright Nanjing Qing So information technology
#########################################################################

from params import Params 
par=Params()
import sys 
sys.path.insert(0,par.caffe_root)
import caffe
from caffe.io import load_image
import os.path as osp
import numpy as np
import skimage
from skimage import transform
from random import shuffle
import copy

class Batchloader(object):
    '''This class abstracts away the loading of images'''
    def __init__(self,imagepath,annotion):
        self._cur=0
        self.imagepath=imagepath
        print "...........batchloader class...."
        with open(annotion,'r') as f:
            self.indexlist=f.readlines()
            shuffle(self.indexlist)

    def get_minibatch(self,phase):
        '''get_minibatch for epoch'''
        blob=[]
        for i in xrange(par.batchsize_dic[phase]):
            data_1,data_2,sim=self.load_next_image()
            temp=[data_1,data_2,sim]
            blob.append(copy.deepcopy(temp))
        return blob


    def load_next_image(self):
        '''load next image in same batchsize'''
        if self._cur==len(self.indexlist):
            self._cur=0
            shuffle(self.indexlist)
        image_1,image_2,sim=self.indexlist[self._cur][:-1]#extract one line and remove '\n'
        data_1=load_image(osp.join(self.imagepath,image_1))
        data_2=load_image(osp.join(self.imagepath,image_2))
        try:
            data_1=self.preprocess(data_1)
            data_2=self.preprocess(data_2)
        except:
            print osp.join(self.imagepath,image_1)
            print osp.join(self.imagepath,image_2)
        self._cur+=1
        return data_1,data_2,sim

    def preprocess(self,image):

        img=transform.resize(image,[par.height,par.width,par.channel])
        img=img.transpose((2,0,1))
        img=img[(2,1,0),:,:]
        img*=255
        mean=np.array([128.0,128.0,128.0])
        mean=mean[:,np.newaxis,np.newaxis]
        img-=mean
        return img
