
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
    logs = './logs/'
    logdir = os.path.join(logs,str(date))
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    else:
        logdir = logdir+"_"+str(np.random.randint(0,1000))
        os.mkdir(logdir)
    
    train_data = AllInOneData(opt.train_path,set='test',transforms=transforms.Compose([Normalizer(),Resizer()]))
    train_generator = torch.utils.data.DataLoader(train_data,batch_size=opt.batch_size,shuffle=True,num_workers=8,
                                                    collate_fn=collater,drop_last=True)

    valid_data = AllInOneData(opt.train_path,set='validation',transforms=transforms.Compose([Normalizer(),Resizer()]))
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
    writer = SummaryWriter(logdir=logdir,filename_suffix=f'Train_{"_".join(opt.heads)}',comment='try1')
    #dummy = torch.ones([1,3,512,512]).cuda()
    #writer.add_graph(model,dummy)
    
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
            
            losses, lm_mask = criterion(out,annot,out['anchors'])
            
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
                      0.4, 0.4)
                
                writer.add_image_with_boxes('prediction',im,pp[0]['rois'],epoch)
                img2 = out['face_landmarks']
                writer.add_images('landmarks prediction',img2,epoch)

                #target_map = torch.zeros((opt.batch_size,1,512,512))
                #target = gt_face_landmarks
                #target = torch.clamp(target,0,511)
                #for b in range(opt.batch_size):
                    #target_map[b,:,target[b,:,:,1].long(),target[b,:,:,0].long()] = 1
                writer.add_images('landmark target', lm_mask,epoch)
                #writer.add_image('target image', im,epoch)


            if epoch%1000==0:
                torch.save(model.state_dict(),f'{logdir}/{opt.save_name.split(".pt")[0]}_epoch_{epoch}.pt')
    torch.save(model.state_dict(),f'{logdir}/{opt.save_name.split(".pt")[0]}_last.pt')
        
             



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',type=str,default='./datasets/Train2021')
    parser.add_argument('--epochs',type=int,default=10000)
    parser.add_argument('--valid_step',type=int,default=1)
    parser.add_argument('--lr',type=int,default=1e-3)
    parser.add_argument('--batch_size',type=int,default=3)
    parser.add_argument('--momentum',type=int,default=0.9)
    parser.add_argument('--wd',type=int,default=0.8)
    parser.add_argument('--optim',type=str,default='Adam')
    parser.add_argument('--save_name',type=str,default='model4.pt')
    parser.add_argument('--heads',nargs='+',default = ["gender","person","face_landmarks"])

    opt = parser.parse_args()
    print(f"Batch size: {opt.batch_size}")
    train(opt)
