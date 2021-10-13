import torch
import torch.nn as nn
import cv2
import numpy as np
from scipy.signal import gaussian
#from utils import BBoxTransform, ClipBoxes
#from utils import postprocess, invert_affine, display
from collections import namedtuple

def calc_iou(a, b):
    # a(anchor) [boxes, (y1, x1, y2, x2)]
    # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]
    a = a.float()
    
    b = b.float()
    
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU


class FocalLoss(nn.Module):
    def __init__(self, device):
        super(FocalLoss, self).__init__()
        self.device = device

    def forward(self, classifications, regressions, anchors, annotations, **kwargs):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        global IoU_argmax

        anchor = anchors[0, :, :]  # assuming all image sizes are the same, which it is
        dtype = anchors.dtype

        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            
            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    
                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = alpha_factor.to(self.device)
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                    
                    bce = -(torch.log(1.0 - classification))
                    
                    cls_loss = focal_weight * bce
                    
                    regression_losses.append(torch.tensor(0).to(dtype).to(self.device))
                    classification_losses.append(cls_loss.sum())
                else:
                    
                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                    
                    bce = -(torch.log(1.0 - classification))
                    
                    cls_loss = focal_weight * bce
                    
                    regression_losses.append(torch.tensor(0).to(dtype))
                    classification_losses.append(cls_loss.sum())

                continue
                
            IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            # compute the loss for classification
            targets = torch.ones_like(classification) * -1
            if torch.cuda.is_available():
                targets = targets.to(self.device)

            targets[torch.lt(IoU_max, 0.4), :] = 0
            global positive_indices
            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones_like(targets) * alpha
            if torch.cuda.is_available():
                alpha_factor = alpha_factor.to(self.device)

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            zeros = torch.zeros_like(cls_loss)
            if torch.cuda.is_available():
                zeros = zeros.to(self.device)
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # efficientdet style
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
                targets = targets.t()

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).to(dtype).to(self.device))
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))

        # debug
        imgs = kwargs.get('imgs', None)
        if imgs is not None:
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            obj_list = kwargs.get('obj_list', None)
            out = postprocess(imgs.detach(),
                              torch.stack([anchors[0]] * imgs.shape[0], 0).detach(), regressions.detach(), classifications.detach(),
                              regressBoxes, clipBoxes,
                              0.5, 0.3)
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
            imgs = ((imgs * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
            imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
            display(out, imgs, obj_list, imshow=False, imwrite=True)

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(regression_losses).mean(dim=0, keepdim=True)

class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        '''
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        '''
       
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

class JointsLoss(nn.Module):
    def __init__(self,device, use_target_weight=True):
        super(JointsLoss, self).__init__()
        self.criterion = AdaptiveWingLoss()
        self.use_target_weight = use_target_weight
        self.k_size=7
        krnl = gaussian(self.k_size,3).reshape(self.k_size,1)
        krnl = np.outer(krnl,krnl)*255
        krnl = torch.from_numpy(krnl).reshape(1,1,self.k_size,self.k_size).type(torch.FloatTensor)
        self.krnl = krnl.to(device)
        self.device = device

    def forward(self, output, target):
        num_joints = target.shape[1]
        target_map = torch.zeros((target.shape[0],1,512,512)).to(self.device)
        target = target
        target = torch.clamp(target,0,511)
        for b in range(target.shape[0]):
            target_map[b,:,target[b,:,:,1].long(),target[b,:,:,0].long()] = 1
        
        # Gausian kernel for heatmap generation!!!
        lm_mask = torch.nn.functional.conv2d(target_map,self.krnl,padding=(self.k_size-1)//2)
        lm_mask = lm_mask/torch.max(lm_mask)
        loss = self.criterion(output,lm_mask)        
        return (loss / num_joints).unsqueeze(0) , lm_mask


class JointsLoss2(nn.Module):
    def __init__(self,device, use_target_weight=True):
        super(JointsLoss2, self).__init__()
        self.criterion = AdaptiveWingLoss()
        self.use_target_weight = use_target_weight
        self.k_size=7
        krnl = gaussian(self.k_size,3).reshape(self.k_size,1)
        krnl = np.outer(krnl,krnl)*255
        krnl = torch.from_numpy(krnl).reshape(1,1,self.k_size,self.k_size).type(torch.FloatTensor)
        self.krnl = krnl.to(device)
        
        self.device = device

    def forward(self, output, target):
        
        num_joints = target.shape[2]
        scale = 512//output.shape[-1]
        self.krnl = self.krnl.repeat(num_joints,num_joints,1,1)
        target_map = torch.zeros((target.shape[0],num_joints,512//scale,512//scale)).to(self.device)
        target = torch.clamp(target,0,511)
        """
        for b in range(target.shape[0]):
            target_map[b,:,(target[b,:,:,1]//scale).long(),(target[b,:,:,0]//scale).long()] = 1
        """
        for b in range(target.shape[0]):
            for j in range(target.shape[2]):
                for p in range(target.shape[1]):
                    target_map[b,j,(target[b,p,j,1]//scale).long(),(target[b,p,j,0]//scale).long()] = 1
        
        # Gausian kernel for heatmap generation!!!
        lm_mask = torch.nn.functional.conv2d(target_map,self.krnl,padding=(self.k_size-1)//2)
        lm_mask = lm_mask/torch.max(lm_mask)
        loss = self.criterion(output,lm_mask)        
        return (loss / 1).unsqueeze(0) , lm_mask




class MTLoss(nn.Module):
    def __init__(self,heads, device):
        super(MTLoss,self).__init__()
        self.heads = heads
        self.total_loss = 0
        self.losses = {key:0 for key in self.heads}
        self.device = device

    def forward(self,pred,annot,anchors):
        pred_person_bbox = pred['person']
        pred_gender = pred['gender']
        person_annot = annot['person']
        gender_annot = annot['gender']
        pg_anot = torch.cat((person_annot.long(),gender_annot.unsqueeze(2)),2)
        gender_loss,person_loss = FocalLoss(device=self.device)(pred_gender,pred_person_bbox,anchors,pg_anot)
        self.losses['gender'] = gender_loss
        self.losses['person'] = person_loss

        if 'face' in self.heads and 'emotions' in self.heads:
            pred_face_bbox = pred['face']
            pred_emotions = pred['emotions']
            face_annot = annot['face']
            emotions_annot = annot['emotions']
            fe_anot = torch.cat((face_annot.long(),emotions_annot.unsqueeze(2)),2)
            emotions_loss,face_loss = FocalLoss(self.device)(pred_emotions,pred_face_bbox,anchors,fe_anot)
            self.losses['emotions'] = emotions_loss
            self.losses['face'] = face_loss
        elif 'face' in self.heads:
            pred_face_bbox = pred['face']
            face_annot = annot['face']
            fe_anot = torch.cat((face_annot.long(),gender_annot.unsqueeze(2)),2)
            emotions_loss,face_loss = FocalLoss(self.device)(pred_gender,pred_face_bbox,anchors,fe_anot)
            self.losses['face'] = face_loss

        if 'face_landmarks' in self.heads:
            pred_lm = pred['face_landmarks']
            lm_anot = annot['face_landmarks']
            lm_loss, lm_mask = JointsLoss2(self.device,False)(pred_lm,lm_anot)
            self.losses['face_landmarks'] = lm_loss*100
        if 'pose' in self.heads:
            pred_pose = pred['pose']
            pose_anot = annot['pose']
            pose_loss = JointsLoss2(self.device,False)(pred_pose,pose_anot)
            self.losses['pose'] = pose_loss


        return self.losses, lm_mask
