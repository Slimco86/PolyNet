import datetime
import os
import argparse
import traceback
from rutines import postprocess
from util import BBoxTransform, ClipBoxes
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import *
from EfficientDet import EfficientDetBackbone , EfficientDetMultiBackbone
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
from dataset import *
from loss import FocalLoss, MTLoss



cap = cv2.VideoCapture(0)
path = 'logs/trained_models/model4_last.pt'
model = EfficientDetMultiBackbone('./datasets/Train2021',compound_coef=0)
model.load_state_dict(torch.load(path),strict=False)
model.cuda()
model.requires_grad_(False)
model.eval()
mean = np.array([0.485, 0.456, 0.406])
std=np.array([0.229, 0.224, 0.225])
img_size = 512
exit = False

while True:
    if exit:
        break
    ret,img = cap.read()
    if ret:
        img = img/np.max(img)
        img = ((img.astype(np.float32) - mean) / std)
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

        new_image = np.zeros((img_size, img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image
        im_ov = new_image*255
        im_ov = im_ov.astype(np.uint8)
        ni = torch.from_numpy(new_image).permute(2,0,1).unsqueeze(0).to(torch.float32).cuda()
        out = model(ni)
        gender = out['gender']
        gender = gender[0].detach().cpu().numpy()
        positive_ind = np.where(gender>0.8)[0]
        person_bbox = out['person'][0].detach().cpu().numpy()
        face_landmarks = out['face_landmarks']
        face_landmarks = face_landmarks.permute(0,2,3,1)
        face_landmarks = face_landmarks[0].detach().cpu().numpy().astype(np.float32)
        face_landmarks = face_landmarks/np.max(face_landmarks)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        pp = postprocess(ni,
                      out['anchors'], out['person'], out['gender'],
                      regressBoxes, clipBoxes,
                      0.8, 0.2)
            
        for box,gen in zip(pp[0]['rois'],pp[0]['class_ids']):
            cv2.rectangle(new_image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),2)
            cv2.putText(new_image,str(gen),(int(box[0]),int(box[1])+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)

        col_map = cv2.applyColorMap((255*face_landmarks).astype(np.uint8),cv2.COLORMAP_JET)
        im_show = cv2.addWeighted(im_ov,0.5,col_map,0.5,0)
        while True and not exit:
            cv2.imshow('Prediction',new_image)
            cv2.imshow('Landmarks',im_show)
            key = cv2.waitKey(10)
            #if key == ord('n'):
            #    break
            if key == ord('q'):
                exit = True
                break
            break
cap.release()
cv2.destroyAllWindows()
        
