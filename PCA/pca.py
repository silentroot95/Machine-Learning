# -*-coding:utf-8 -*-
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
#忽略警告
warnings.simplefilter('ignore')
#这两行解决图例中文乱码问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

im = Image.open(r'.\pic\2.jpeg')
#转为灰度图
im2 = im.convert('L')
#转为矩阵
arr2 = np.array(im2)

class PCA:
    def __init__(self,data):
        #原始图片数据
        self.data = data
        #行列像素
        self.row,self.col=data.shape
    def __pca(self,fvec,k):
        '''
        pca 主成分分析
        fmat为数据集组成的矩阵,每列是一个数据
        k   为前k个主成分
        '''
        #数据均值
        mvec = fvec.mean(axis=1)
        row,col = fvec.shape
        #均值化
        bias = np.array([x-mvec for x in fvec.T])
        #协方差矩阵
        cov = np.matmul(bias.T,bias)/col
        #特征值与特征向量
        val,vec = np.linalg.eig(cov)
        #平方和失真率
        rate = sum(val[k:])/sum(val)
        #压缩率  压缩后的数据量/原图片的数据量
        com_rate = (k*col+row+k*row)/(self.row*self.col)
        #特征向量前k个
        kvec = vec[:,:k]
        #bias*kvec*kvec的转置为新坐标系下的均值化误差
        tmp = np.matmul(bias,kvec)
        #新坐标系下的均值化误差
        cbias = np.matmul(tmp,kvec.T)
        #加上原来的均值
        ans = np.array([x+mvec for x in cbias])
        #转为uint8类型0-255
        return np.uint8(ans.T),com_rate
    def col_pca(self,k):
        '''
        按列主成分分析
        上面的__pca函数就是按列进行的
        '''
        fvec_pca,com_rate= self.__pca(self.data,k)
        return fvec_pca,com_rate
    def row_pca(self,k):
        '''
        按行主成分分析
        先转置，分析之后，再转置
        '''
        fvec = self.data.T
        fvec_pca,com_rate = self.__pca(fvec,k)
        return fvec_pca.T,com_rate
    def tb_pca(self,k,n):
        '''
        按块主成分分析，块的大小为n*n
        块就是一个滑动窗口，除行或列的最后一个窗口可能（当图片大小无法整除块时）重叠外，每个窗口无重叠
        k   前k个特征向量
        '''
        fvec=[]
        r=c=0
        while r+n<=self.row:
            c=0
            while c+n<=self.col:
                #取窗口，然后转为向量
                f = self.data[r:r+n,c:c+n].flatten()
                fvec.append(f)
                c+=n
            if self.col%n != 0:
                fvec.append(self.data[r:r+n,(self.col-n):].flatten())
            r+=n
            if 0<self.row-r<n:
                r = self.row-n
        fvec = np.array(fvec).T
        #主成分分析
        fvec_pca,com_rate= self.__pca(fvec,k)
        fvec_pca = fvec_pca.T
        r=i=c=0
        rec = np.zeros(self.data.shape)
        while(r+n<=self.row):
            c=0
            while(c+n<=self.col):
                #根据向量转化为窗口，重构图片
                rec[r:r+n,c:c+n] = fvec_pca[i].reshape((n,n))
                i+=1
                c+=n
                #列无法整除时，添加最后一个窗口
            if self.col%n != 0:
                rec[r:r+n,self.col-n:]=fvec_pca[i].reshape((n,n))
                i+=1
            r+=n
                #行无法整除时
            if 0<self.row-r<n:
                r = self.row-n
        return rec,com_rate

def RGB_PCA(im,mode):
    '''
    彩色图片pca
    按块pca效果较好
    r,g,b单通道pca后再叠加
    '''
    t =[]
    for co in im.split():
        co = np.array(co)
        copca = PCA(co)
        if mode == 'c':
            cod,com_rate = copca.col_pca(33)
        elif mode == 'r':
            cod,com_rate = copca.row_pca(33)
        elif mode == 'b':
            cod,com_rate = copca.tb_pca(1,2)
        co_img = Image.fromarray(np.uint8(cod))
        t.append(co_img)
    t_img = Image.merge('RGB',t)
    return t_img,com_rate
def L_main():
    '''
    灰度图压缩步骤
    '''
    dp = PCA(arr2)
    colpca,crate= dp.col_pca(33)
    rowpca,rrate = dp.row_pca(33)
    tbpca,tbrate= dp.tb_pca(1,2)
    #img_pca = Image.fromarray(np.uint8(dpca))
    #plt.imshow(img_pca)
    plt.subplot(221)
    plt.title('灰度图')
    plt.imshow(arr2,cmap='gray')
    plt.axis('off')
    plt.subplot(222)
    plt.title('按列压缩，压缩率%.2f' %crate)
    plt.imshow(colpca,cmap='gray')
    plt.axis('off')
    plt.subplot(223)
    plt.title('按行压缩，压缩率%.2f' %rrate)
    plt.imshow(rowpca,cmap='gray')
    plt.axis('off')
    plt.subplot(224)
    plt.title('按块压缩，压缩率%.2f' %tbrate)
    plt.imshow(tbpca,cmap='gray')
    plt.axis('off')
    plt.show()
def RGB_main():
    '''
    RGB彩图压缩步骤
    '''
    cimg,ccom_rate= RGB_PCA(im,'c')
    rimg,rcom_rate= RGB_PCA(im,'r')
    bimg,bcom_rate= RGB_PCA(im,'b')
    plt.subplot(221)
    plt.imshow(im)
    plt.title('原图')
    plt.axis('off')
    plt.subplot(222)
    plt.title('按列压缩，压缩率%.2f' %ccom_rate)
    plt.imshow(cimg)
    plt.axis('off')
    plt.subplot(223)
    plt.title('按行压缩，压缩率%.2f' %rcom_rate)
    plt.imshow(rimg)
    plt.axis('off')
    plt.subplot(224)
    plt.title('按块压缩，压缩率%.2f' %bcom_rate)
    plt.imshow(bimg)
    plt.axis('off')
    plt.show()
if __name__ == '__main__':
    L_main()
    RGB_main()
