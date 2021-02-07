
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
from datetime import datetime

def train(opt):
    date = datetime.date(datetime.now())
    logs = '../logs/'
    logdir = os.path.join(logs,str(date))
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    else:
        logdir = logdir+"_"+str(np.random.randint(0,1000))
        os.mkdir(logdir)
    
    train_data = AllInOneData(opt.train_path,set='train',transforms=transforms.Compose([Normalizer(),Resizer()]))
    train_generator = torch.utils.data.DataLoader(train_data,batch_size=opt.batch_size,shuffle=True,num_workers=8,
                                                    collate_fn=collater,drop_last=True)

    valid_data = AllInOneData(opt.train_path,set='validation',transforms=transforms.Compose([Normalizer(),Resizer()]))
    valid_generator = torch.utils.data.DataLoader(valid_data,batch_size=opt.batch_size,shuffle=False,num_workers=8,
                                                    collate_fn=collater,drop_last=True)
    
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model = EfficientDetMultiBackbone(opt.train_path,compound_coef=0,heads=opt.heads)
    model.to(device)

    min_val_loss = 10e5
    
    if opt.optim == 'Adam':
        optimizer = torch.optim.AdamW(model.parameters(),lr=opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(),lr=opt.lr,momentum = opt.momentum,nesterov=True)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, opt.lr, total_steps=None, epochs=opt.epochs,
                                                    steps_per_epoch=len(train_generator), pct_start=0.1, anneal_strategy='cos',
                                                    cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, 
                                                    div_factor=25.0, final_div_factor=1000.0, last_epoch=-1)

    criterion = MTLoss(heads = opt.heads, device = device)
    
    print('Model is successfully initiated')
    print(f'Targets are {opt.heads}.')
    verb_loss = 0
    writer = SummaryWriter(logdir=logdir,filename_suffix=f'Train_{"_".join(opt.heads)}',comment='try1')
    
    for epoch in range(opt.epochs):
        model.train()
        Losses = {k:[] for k in opt.heads}
        description = f'Epoch:{epoch}| Total Loss:{verb_loss}'
        progress_bar = tqdm(train_generator,desc = description)
        Total_loss = []
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

            out = model(imgs)
            annot = {'person':gt_person_bbox,'gender':gt_gender,
                     'face':gt_face_bbox,'emotions':gt_emotions,
                     'face_landmarks':gt_face_landmarks,
                     'pose':gt_pose}
            
            losses, lm_mask = criterion(out,annot,out['anchors'])
            loss = torch.zeros(1).to(device)
            loss = torch.sum(torch.cat(list(losses.values())))
            loss.backward()
            optimizer.step()
            scheduler.step() 

            verb_loss = loss.detach().cpu().numpy()
            Total_loss.append(verb_loss)
            description = f'Epoch:{epoch}| Total Loss:{verb_loss}|'
            for k,v in losses.items():
                Losses[k].append(v.detach().cpu().numpy())
                description+=f'{k}:{round(np.mean(Losses[k]),1)}|'
            progress_bar.set_description(description)
            optimizer.zero_grad()
        
        writer.add_scalar('Train/Total',round(np.mean(Total_loss),2),epoch)
        for k in Losses.keys():
            writer.add_scalar(f"Train/{k}",round(np.mean(Losses[k]),2),epoch)
           
        if epoch%opt.valid_step==0:
            im = imgs[0]
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            pp = postprocess(imgs,
                  out['anchors'], out['person'], out['gender'],
                  regressBoxes, clipBoxes,
                  0.4, 0.4)
            
            writer.add_image_with_boxes('Train/Box_prediction',im,pp[0]['rois'],epoch)
            img2 = out['face_landmarks']
            writer.add_images('Train/landmarks_prediction',img2,epoch)
            writer.add_images('Train/landmark target', lm_mask,epoch)
            
            #VALIDATION STEPS
            model.eval()
            with torch.no_grad():
                valid_Losses = {k:[] for k in opt.heads}

                val_description = f'Validation| Total Loss:{verb_loss}'
                progress_bar = tqdm(valid_generator,desc = val_description)
                Total_loss = []
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
                    out = model(imgs)
                    annot = {'person':gt_person_bbox,'gender':gt_gender,
                     'face':gt_face_bbox,'emotions':gt_emotions,
                     'face_landmarks':gt_face_landmarks,
                     'pose':gt_pose}

                    losses, lm_mask = criterion(out,annot,out['anchors'])

                    loss = torch.zeros(1).to(device)
                    loss = torch.sum(torch.cat(list(losses.values())))
                    verb_loss = loss.detach().cpu().numpy()
                    Total_loss.append(verb_loss)
                    val_description = f'Validation| Total Loss:{verb_loss}|'
                    for k,v in losses.items():
                        valid_Losses[k].append(v.detach().cpu().numpy())
                        val_description+=f'{k}:{round(np.mean(valid_Losses[k]),1)}|'
                    progress_bar.set_description(val_description)

                writer.add_scalar('Validation/Total',round(np.mean(Total_loss),2),epoch)
                for k in valid_Losses.keys():
                    writer.add_scalar(f"Validation/{k}",round(np.mean(valid_Losses[k]),2),epoch)

                im = imgs[0]
                regressBoxes = BBoxTransform()
                clipBoxes = ClipBoxes()
                pp = postprocess(imgs,
                  out['anchors'], out['person'], out['gender'],
                  regressBoxes, clipBoxes,
                  0.4, 0.4)

                writer.add_image_with_boxes('Validation/Box_prediction',im,pp[0]['rois'],epoch)
                img2 = out['face_landmarks']
                writer.add_images('Validation/landmarks_prediction',img2,epoch)
                writer.add_images('Validation/landmark target', lm_mask,epoch)

                if verb_loss<min_val_loss:
                    print("The model improved and checkpoint is saved.")
                    torch.save(model.state_dict(),f'{logdir}/{opt.save_name.split(".pt")[0]}_best_epoch_{epoch}.pt')
                    min_val_loss = verb_loss
                

        if epoch%100==0:
            torch.save(model.state_dict(),f'{logdir}/{opt.save_name.split(".pt")[0]}_epoch_{epoch}.pt')
    torch.save(model.state_dict(),f'{logdir}/{opt.save_name.split(".pt")[0]}_last.pt')
    writer.close()
        
             



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',type=str,default='/home/ROSENINSPECTION/ivoloshenko/dataset/Train2021')
    parser.add_argument('--epochs',type=int,default=5000)
    parser.add_argument('--valid_step',type=int,default=10)
    parser.add_argument('--lr',type=int,default=1e-3)
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--momentum',type=int,default=0.9)
    parser.add_argument('--wd',type=int,default=0.8)
    parser.add_argument('--optim',type=str,default='Adam')
    parser.add_argument('--save_name',type=str,default='model.pt')
    parser.add_argument('--heads',nargs='+',default = ["gender","person","face_landmarks"])

    opt = parser.parse_args()
    print(f"Batch size: {opt.batch_size}")
    train(opt)
