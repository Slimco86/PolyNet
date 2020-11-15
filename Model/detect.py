import datetime
import os
import argparse
import traceback

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
model = EfficientDetMultiBackbone(0)
model.init_backbone('logs/trained_models/best_model4.pt')
model.freeze_bn()
model.cuda()
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
        ni = torch.from_numpy(new_image).permute(2,0,1).unsqueeze(0).to(torch.float32).cuda()
        features, person_bbox, face_bbox, face_landmarks, pose,age, gender,race, skin, emotion, anchors = model(ni)
        race = race[0].detach().cpu().numpy()
        positive_ind = np.where(race>0.65)[0]

        if positive_ind.shape[0]>0:
            face_bbox = face_bbox[0,positive_ind].detach().cpu().numpy()
            face_landmarks = face_landmarks[0,positive_ind].detach().cpu().numpy().astype(np.int64)
            age = age[0,positive_ind].detach().cpu().numpy().astype(np.int64)

            
            for box,lm,ag in zip(face_bbox.astype(np.int64),face_landmarks,age):
                cv2.rectangle(new_image,(box[0],box[1]),(box[2],box[3]),(0,0,255),2)
                cv2.putText(new_image,str(ag),(box[0],box[1]+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
                for i in range(0,lm.shape[0],2):
                    cv2.circle(new_image,(lm[i],lm[i+1]),3,(150,0,255),cv2.FILLED)

        while True and not exit:
            cv2.imshow('Prediction',new_image)
            key = cv2.waitKey(10)
            #if key == ord('n'):
            #    break
            if key == ord('q'):
                exit = True
                break
            break
cap.release()
cv2.destroyAllWindows()
        
