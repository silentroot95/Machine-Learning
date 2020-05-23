#!/usr/bin/python
#-*- coding:utf-8 -*-
'''
Created on May 23,2020
Author: silentroot95
'''
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

filename = r'.\1.jpg'
def LoadData(filename):
    im = Image.open(filename)
    im2 = im.convert('L')
    arr2 = np.mat(im2)
    return arr2
def SVD(data_mat,k):
    u,sigma,vT = np.linalg.svd(data_mat)
    uk = u[:,0:k]
    sk = np.diag(sigma[:k])
    info_rate = sum(sigma[:k])/sum(sigma)
    vTk = vT[0:k,:]
    recon = uk*sk*vTk
    return recon,info_rate
def main():
    data = LoadData(filename)
    k =20
    m,n = data.shape
    com_rate = k*(m+n+1)/(m*n)
    recon,info_rate = SVD(data,k)
    yt = Image.fromarray(data)
    im_svd1 = Image.fromarray(recon)
    plt.subplot(121)
    plt.title('灰度图')
    plt.imshow(yt,cmap = 'gray')
    plt.axis('off')
    plt.subplot(122)
    plt.title('压缩率{0:.2f},信息率{1:.2f}'.format(com_rate,info_rate))
    plt.imshow(im_svd1,cmap = 'gray')
    plt.axis('off')
    plt.show()
if __name__ =='__main__':
    main()
