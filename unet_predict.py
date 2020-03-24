###################预测#########################
#predict预测整张大的遥感影像

import cv2
import random
import numpy as np
import os
import argparse
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TEST_SET = ['1.png','2.png','3.png']

image_size = 256

classes = [0. ,  1.,  2.,   3.  , 4.]  
  
labelencoder = LabelEncoder()  
labelencoder.fit(classes) 

predict_path='./predict'
if not os.path.exists(predict_path):
    os.makedirs(predict_path)

def args_parse():
# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="Trained_Unet_Model.h5",
        help="name to trained model ")
    ap.add_argument("-s", "--stride", type=int, default=image_size,
        help="crop slide stride")  # 滑动步长
    args = vars(ap.parse_args())    
    return args

    
def predict(args):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(args["model"])

    stride = args['stride']
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        #load the image
        image = cv2.imread('./test/' + path)

        # calculate the value of padding
        h,w,_ = image.shape
        padding_h = (h//stride + 1) * stride    # " // "表示整数除法
        padding_w = (w//stride + 1) * stride
        padding_img = np.zeros((padding_h,padding_w,3),dtype=np.uint8)
        padding_img[0:h,0:w,:] = image[:,:,:]
        padding_img = padding_img.astype("float") / 255.0
        padding_img = img_to_array(padding_img)   # 从(h,w,c)到(c,h,w)
        print('{}:{}'.format(path,padding_img.shape))

        # take out the small image to predict
        mask_whole = np.zeros((padding_h,padding_w),dtype=np.uint8)
        for i in range(padding_h//stride):
            for j in range(padding_w//stride):
                # 小图左上角坐标(Li,Lj )
                Li,Lj = (i*stride,j*stride)
                crop = padding_img[:3,Li:Li+image_size,Lj:Lj+image_size]   # small image
                _,ch,cw = crop.shape
                if ch != 256 or cw != 256:
                    print('invalid size!')
                    continue
                # predeict the small image
                crop = np.expand_dims(crop, axis=0) # (c,h,w)--->(1,c,h,w)
                pred = model.predict(crop,verbose=2)
                # print (np.unique(pred))
                pred = pred.reshape((256,256)).astype(np.uint8)
                #print 'pred:',pred.shape
                mask_whole[Li:Li+image_size,Lj:Lj+image_size] = pred[:,:]

        # save the final result
        cv2.imwrite(predict_path+"/"+path[:-4]+'.png',mask_whole[0:h,0:w])
        
    

    
if __name__ == '__main__':
    args = args_parse()
    predict(args)
    print("predict process over!")



