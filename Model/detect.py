import datetime
import os
import argparse
from rutines import postprocess
from util import BBoxTransform, ClipBoxes
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import *
from EfficientDet import EfficientDetBackbone , EfficientDetMultiBackbone
import numpy as np
from dataset import *
import time



Genders = ["Male", "Female", "unknown"]
lm_threshold = 0.05
cap = cv2.VideoCapture("/home/rosen.user/Videos/SPOON.mp4")
cap.set(1,52*24)
path = 'logs/2021-01-30/model_epoch_3000.pt'
model = EfficientDetMultiBackbone('./datasets/Train2021',compound_coef=0)

if torch.cuda.is_available():
    model.load_state_dict(torch.load(path,map_location=torch.device('cuda:0')),strict=False)
    model.cuda()
else:
    model.load_state_dict(torch.load(path,map_location=torch.device('cpu')),strict=False)
model.requires_grad_(False)
model.eval()
mean = np.array([0.66, 0.58, 0.64])
std=np.array([0.22, 0.22, 0.23])
img_size = 512
exit = False
i = 0 
while True:
    start = time.time()
    if exit:
        break
    ret,img = cap.read()
    #img= img[:,0:672,:] for stereolabs
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    if ret:
        img = img.astype(np.float64)/np.max(img)
        img = (img - mean) / std
        height, width, _ = img.shape
        if height > width:
            scale = img_size / height
            resized_height = img_size
            resized_width = int(width * scale)
        else:
            scale = img_size / width
            resized_height = int(height * scale)
            resized_width = img_size

        image = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((img_size, img_size, 3),dtype=np.float32)
        new_image[0:resized_height, 0:resized_width] = image

        ni = torch.from_numpy(new_image).permute(2,0,1).unsqueeze(0).to(torch.float32)
        if torch.cuda.is_available():
            out = model(ni.cuda())
        else:
            out = model(ni)
        gender = out['gender']
        gender = gender[0].detach().cpu().numpy()
        positive_ind = np.where(gender>0.8)[0]
        person_bbox = out['person'][0].detach().cpu().numpy()
        face_landmarks = out['face_landmarks']
        face_landmarks = face_landmarks.permute(0,2,3,1)
        face_landmarks = face_landmarks[0].detach().cpu().numpy().astype(np.float32)
        if np.max(face_landmarks) < lm_threshold:
            face_landmarks = np.zeros((512,512))
        else:
            face_landmarks = face_landmarks/np.max(face_landmarks)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        pp = postprocess(ni,
                      out['anchors'], out['person'], out['gender'],
                      regressBoxes, clipBoxes,
                      0.8, 0.2)
        end = time.time() 
        disp_img = new_image*std+mean
        im_ov = disp_img*255
        im_ov = im_ov.astype(np.uint8)
        im_ov = cv2.cvtColor(im_ov,cv2.COLOR_BGR2RGB)
        
        for box,gen in zip(pp[0]['rois'],pp[0]['class_ids']):
            cv2.rectangle(im_ov,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),2)
            cv2.putText(im_ov,str(Genders[gen]),(int(box[0]),int(box[1])+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(im_ov,f"FPS:{round(1/(end-start),1)}",(10,500),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)

        col_map = cv2.applyColorMap((255*face_landmarks).astype(np.uint8),cv2.COLORMAP_JET)
        im_show = cv2.addWeighted(im_ov,0.5,col_map,0.5,0)

        while True and not exit:
            cv2.imshow('Prediction',im_ov)
            cv2.imshow('Landmarks',im_show)
            save_img = np.zeros((512,1024,3))
            save_img[:,:512,:] = im_ov
            save_img[:,512:,:] = im_show
            cv2.imwrite(f'fr{i}.jpg',save_img,)
            i+=1
            key = cv2.waitKey(10)
            #if key == ord('n'):
            #    break
            if key == ord('q'):
                exit = True
                break
            break
cap.release()
cv2.destroyAllWindows()
        
