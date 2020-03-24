#coding=utf-8

#########################introduction##################
"""执行该文件gen_dataset.py，对大图片进行切割，生成unet训练集"""
import cv2
import random
import os
import numpy as np
from tqdm import tqdm   #进度条
import matplotlib.pyplot as plt # plt 用于显示图片

# 全局变量
img_w = 256  
img_h = 256

image_sets = ['1.png','2.png','3.png','4.png','5.png']

visualize_path="./unet_train/visualize"
src_roi_path="./unet_train/road/src"
label_roi_path="./unet_train/road/label"


def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)
    

def rotate(xb,yb,angle):
    #cv2.getRotationMatrix2D()，这个函数需要三个参数，旋转中心，旋转角度，旋转后图像的缩放比例
    M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
    #cv2.warpAffine()仿射变换，参数src - 输入图像  M - 变换矩阵(一般反映平移或旋转的关系)  dsize - 输出图像的大小  flags - 插值方法的组合 。。。
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb,yb
    
def blur(img):
    # cv2.blur(均值滤波)，参数说明：img表示输入的图片， (3, 3) 表示进行均值滤波的方框大小
    # 均值滤波是用每个像素和它周围像素计算出来的平均值替换图像中每个像素
    img = cv2.blur(img, (3, 3));
    return img

def add_noise(img):
    for i in range(200): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])  #h
        temp_y = np.random.randint(0,img.shape[1])  #w
        img[temp_x][temp_y] = 255
    return img
    
    
def data_augment(xb,yb):
    # plt.imshow(xb)
    # plt.show()
    # cv2.imshow('src_roi', xb)  # 加载过程特别慢
    # 旋转变换90、180、270
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,90)
        # cv2.imshow('after_src_roi', xb)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,180)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,270)

    # 图像翻转
    if np.random.random() < 0.25:
        # cv2.flip(),第二个参数：1水平翻转  0垂直翻转  -1水平垂直翻转
        xb = cv2.flip(xb, 1)
        yb = cv2.flip(yb, 1)


    # 模糊操作
    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb,1.0)

    # 均值滤波
    if np.random.random() < 0.25:
        xb = blur(xb)

    # 增加噪音
    if np.random.random() < 0.2:
        xb = add_noise(xb)
    # plt.imshow(xb)
    # plt.show()

    return xb,yb

def check_save_Path(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def creat_dataset(image_num = 50000, mode = 'original'):
    print('creating dataset...')
    check_save_Path([visualize_path,src_roi_path,label_roi_path])
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in tqdm(range(len(image_sets))):   #进度条显示：0%|          | 0/5 [00:00<?, ?it/s]
        count = 0
        src_img = cv2.imread('./data/src/' + image_sets[i])  # 3 channels
        label_img = cv2.imread('./data/label/' + image_sets[i],cv2.IMREAD_GRAYSCALE)  # single channel
        X_height,X_width,_ = src_img.shape
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w,:]
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
            # 随机的切割的图像进行数据的增强操作
            if mode == 'augment':
                src_roi,label_roi = data_augment(src_roi,label_roi)
            
            visualize = np.zeros((256,256)).astype(np.uint8)
            visualize = label_roi *50

            cv2.imwrite(visualize_path+'/%d.png' % g_count,visualize)
            cv2.imwrite(src_roi_path+'/%d.png' % g_count,src_roi)
            cv2.imwrite(label_roi_path+'/%d.png' % g_count,label_roi)
            count += 1 
            g_count += 1


if __name__=='__main__':  
    creat_dataset(mode='augment')
    print("over gen_dataset!")
