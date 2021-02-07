from util import ClipBoxes, BBoxTransform
from rutines import postprocess
import torch
import os 
import cv2
import numpy as np
from EfficientDet import EfficientDetMultiBackbone
from dataset import *



if __name__=='__main__':
    
    path = 'logs/2021-01-30/model_epoch_3300.pt'
    model = EfficientDetMultiBackbone('./datasets/Train2021',compound_coef=0)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(path),strict=False)
        model.cuda()
        model.requires_grad_(False)
        model.eval()
    else:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')),strict=False)
        model.requires_grad_(False)
        model.eval()
    
    
    mean = np.array([0.66, 0.58, 0.54])
    std=np.array([0.22, 0.22, 0.23])
    ds = AllInOneData('./datasets/Train2021',set='test',transforms = transforms.Compose([Normalizer(),Resizer()]))
    loader = DataLoader(ds,batch_size=1,shuffle=True,collate_fn=collater)
    batch = next(iter(loader))
    img = batch['img'][0].permute(1,2,0)
    prim = batch['img']
    img = img.numpy()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    im_ov = img*std+mean
    im_ov = im_ov*255
    im_ov = im_ov.astype(np.uint8)
    person = batch['person_bbox'][0].numpy().astype(np.int64)
    landmarks = batch['face_landmarks'][0].numpy().astype(np.int64)
    gender = batch['gender'][0].numpy()
   
    
    for box,lm,ag in zip(person,landmarks,gender):
        cv2.rectangle(im_ov,(box[0],box[1]),(box[2],box[3]),(0,255,255),2)
        cv2.putText(im_ov,str(ag),(box[0],box[1]+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2,cv2.LINE_AA)
        #for i in range(0,lm.shape[0],2):
        #    cv2.circle(img,(lm[i],lm[i+1]),3,(0,0,255),cv2.FILLED)


    if torch.cuda.is_available():
        prim.cuda()
    out = model(prim)
    person_bbox = out['person']
    face_landmarks = out['face_landmarks']
    gender = out['gender']
    gender = gender[0].detach().cpu().numpy()
    positive_ind = np.where(gender>0.8)[0]

    gender = gender[positive_ind]
    person_bbox = person_bbox[0,positive_ind].detach().cpu().numpy().astype(np.float32)*512
    face_landmarks = face_landmarks.permute(0,2,3,1)
    face_landmarks = face_landmarks[0].detach().cpu().numpy().astype(np.float32)
    face_landmarks = face_landmarks/np.max(face_landmarks)
    
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    pp = postprocess(prim,
                      out['anchors'], out['person'], out['gender'],
                      regressBoxes, clipBoxes,
                      0.8, 0.2)

    for box,gen in zip(pp[0]['rois'],pp[0]['class_ids']):
            cv2.rectangle(im_ov,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),2)
            cv2.putText(im_ov,str(gen),(int(box[0]),int(box[1])+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
    
    col_map = cv2.applyColorMap((255*face_landmarks).astype(np.uint8),cv2.COLORMAP_JET)
    im_show = cv2.addWeighted(im_ov,0.5,col_map,0.5,0)
    while True:
        cv2.imshow('test',im_ov)
        cv2.imshow('landmarks',im_show)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()