import torch
import os 
import cv2
import numpy as np
from EfficientDet import EfficientDetMultiBackbone
from dataset import *



if __name__=='__main__':
    
    path = './logs/trained_models/best_model4.pt'
    model = model = EfficientDetMultiBackbone(compound_coef=0)
    model.load_state_dict(torch.load(path))
    model.cuda()
    model.requires_grad_(False)
    model.eval()
    
    ds = AllInOneData('./datasets/Train2020',set='validation',transforms = transforms.Compose([Normalizer(),Resizer()]))
    loader = DataLoader(ds,batch_size=1,shuffle=True,collate_fn=collater)
    batch = next(iter(loader))
    img = batch['img'][0].permute(1,2,0)
    prim = batch['img']
    img = img.numpy()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #print(img.shape)
    face = batch['face_bbox'][0].numpy().astype(np.int64)
    landmarks = batch['face_landmarks'][0].numpy().astype(np.int64)
    age = batch['age'][0].numpy()
    #print(age)
    #print(landmarks.shape)
    for box,lm,ag in zip(face,landmarks,age):
        cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,0,255),2)
        cv2.putText(img,str(ag),(box[0],box[1]+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
        for i in range(0,lm.shape[0],2):
            cv2.circle(img,(lm[i],lm[i+1]),3,(0,0,255),cv2.FILLED)


    features, person_bbox, face_bbox, face_landmarks, pose,age, gender,race, skin, emotion, anchors = model(prim.cuda())
    race = race[0].detach().cpu().numpy()
    positive_ind = np.where(race>0.5)[0]
    print(positive_ind)
    face_bbox = face_bbox[0,positive_ind].detach().cpu().numpy().astype(np.int32)
    face_landmarks = face_landmarks[0,positive_ind].detach().cpu().numpy().astype(np.int32)
    age = age[0,positive_ind].detach().cpu().numpy().astype(np.int32)

    for box,lm,ag in zip(face_bbox,face_landmarks,age):
        cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(255,0,0),2)
        cv2.putText(img,str(ag),(box[0],box[1]+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2,cv2.LINE_AA)
        for i in range(0,lm.shape[0],2):
            cv2.circle(img,(lm[i],lm[i+1]),3,(255,0,0),cv2.FILLED)
    
    while True:
        cv2.imshow('test',img)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()