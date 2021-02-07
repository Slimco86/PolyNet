import cv2
import numpy as np
import os 
import json

def findNorm(path):
    files = os.listdir(path)
    BGR_mean = []
    BGR_std = []
    for file in files:
        if file.endswith('.jpg'):
            img = cv2.imread(os.path.join(path,file))
            w,h = img.shape[0],img.shape[1]
            n_pixels = w*h
            bm = np.sum(img[:,:,0])/n_pixels
            gm = np.sum(img[:,:,1])/n_pixels
            rm = np.sum(img[:,:,2])/n_pixels
            BGR_mean.append([bm,gm,rm])
            bstd = np.sqrt(np.sum((img[:,:,0]-bm)**2)/n_pixels)
            gstd = np.sqrt(np.sum((img[:,:,1]-gm)**2)/n_pixels)
            rstd = np.sqrt(np.sum((img[:,:,2]-rm)**2)/n_pixels)
            BGR_std.append([bstd,gstd,rstd])
    std = np.mean(np.array(BGR_std).reshape((-1,3)),axis=0)/255
    mean = np.mean(np.array(BGR_mean).reshape((-1,3)),axis=0)/255
    print(mean,std)


path = "/home/rosen.user/Documents/Smart_Mirror_rework/datasets/Train2021/validation"
findNorm(path)