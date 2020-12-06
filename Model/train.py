
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
#from util.sync_batchnorm import patch_replication_callback
#from util import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights


def train(opt):
    
    train_data = AllInOneData(opt.train_path,set='test',transforms=transforms.Compose([Normalizer(),Resizer()]))
    train_generator = torch.utils.data.DataLoader(train_data,batch_size=opt.batch_size,shuffle=True,num_workers=8,
                                                    collate_fn=collater,drop_last=True)

    valid_data = AllInOneData(opt.train_path,set='test',transforms=transforms.Compose([Normalizer(),Resizer()]))
    valid_generator = torch.utils.data.DataLoader(valid_data,batch_size=opt.batch_size,shuffle=False,num_workers=8,
                                                    collate_fn=collater,drop_last=True)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = EfficientDetMultiBackbone(opt.train_path,compound_coef=0,heads=opt.heads)
    model.to(device)

    min_val_loss = None

    if opt.optim == 'Adam':
        optimizer = torch.optim.AdamW(model.parameters(),lr=opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(),lr=opt.lr,momentum = opt.momentum,nesterov=True)

    criterion = MTLoss(heads = opt.heads)
    
    print('Model is successfully initiated')
    print(f'Targets are {opt.heads}.')
    verb_loss = 0
    writer = SummaryWriter(logdir='logs',filename_suffix=f'Train_{"_".join(opt.heads)}',comment='try1')
    for epoch in range(opt.epochs):
        model.train()
        Losses = {k:0 for k in opt.heads}
        description = f'Epoch:{epoch}| Total Loss:{verb_loss}'
        progress_bar = tqdm(train_generator,desc = description)
        
        for sample in progress_bar:
                        
            imgs = sample['img'].to(device)
            gt_person_bbox = sample['person_bbox'].to(device)
            gt_face_bbox = sample['face_bbox'].to(device)
            gt_pose = sample['pose'].to(device)
            gt_face_landmarks = sample['face_landmarks'].to(device)
            gt_age = sample['age'].to(device)
            gt_race = sample['race'].to(device)
            gt_gender = sample['gender'].to(device)
            gt_skin = sample['skin'].to(device)
            gt_emotions = sample['emotion'].to(device)
            
            optimizer.zero_grad()
            out = model(imgs)
            annot = {'person':gt_person_bbox,'gender':gt_gender,
                     'face':gt_face_bbox,'emotions':gt_emotions,
                     'face_landmarks':gt_face_landmarks,
                     'pose':gt_pose}
            
            losses = criterion(out,annot,out['anchors'])
            
            loss = torch.zeros(1).to(device)
            loss = torch.sum(torch.cat(list(losses.values())))
            loss.backward()
            optimizer.step()
            verb_loss = loss.detach().cpu().numpy()
            writer.add_scalar('Total',verb_loss,epoch)
            description = f'Epoch:{epoch}| Total Loss:{verb_loss}|'
            for k,v in losses.items():
                Losses[k]+=v.detach().cpu().numpy()
                writer.add_scalar(k,v.detach().cpu().numpy(),epoch)
                description+=f'{k}:{round(np.mean(Losses[k]),1)}|'
            progress_bar.set_description(description)
            
            if epoch%100==0:
                im = imgs[0]
                regressBoxes = BBoxTransform()
                clipBoxes = ClipBoxes()
                pp = postprocess(imgs,
                      out['anchors'], out['person'], out['gender'],
                      regressBoxes, clipBoxes,
                      0.3, 0.3)
                
                writer.add_image_with_boxes('prediction',im,pp[0]['rois'],epoch)
                img2 = out['face_landmarks'].permute(0,2,3,1)
                img2 = img2[0].detach().cpu().numpy()
                img2 = cv2.resize(img2,(512,512))
                img2 = torch.from_numpy(img2).unsqueeze(0)
                writer.add_image('landmarks prediction',img2,epoch)

                target_map = torch.zeros((opt.batch_size,1,256,256))
                target = gt_face_landmarks/2
                for b in range(opt.batch_size):
                    target_map[b,:,target[b,:,:,1].long(),target[b,:,:,0].long()] = 1
                writer.add_image('landmark target', target_map[0],epoch)
                writer.add_image('target image', im,epoch)



        #torch.save(model.state_dict(),f'./logs/trained_models/{opt.save_name.split(".pt")[0]}_{epoch}.pt')
        
        
             



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',type=str,default='./datasets/Train2020')
    parser.add_argument('--epochs',type=int,default=2000)
    parser.add_argument('--valid_step',type=int,default=1)
    parser.add_argument('--lr',type=int,default=1e-4)
    parser.add_argument('--batch_size',type=int,default=3)
    parser.add_argument('--momentum',type=int,default=0.9)
    parser.add_argument('--wd',type=int,default=0.8)
    parser.add_argument('--optim',type=str,default='Adam')
    parser.add_argument('--save_name',type=str,default='model4.pt')
    parser.add_argument('--heads',nargs='+',default = ["gender","person","face_landmarks"])

    opt = parser.parse_args()
    print(f"Batch size: {opt.batch_size}")
    train(opt)
