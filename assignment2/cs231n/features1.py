import numpy as np
import cv2

def hist(imgs):
    imgs_features=np.zeros_like(imgs)   
    num_images = imgs.shape[0]    
    for i in xrange(num_images):    
        for j in range(3):        
            temp=imgs[i,j,:,:].astype('uint8')
            imgs_features[i,j,:,:]=cv2.equalizeHist(temp)
    return imgs_features