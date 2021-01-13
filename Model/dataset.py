import torch
import os
import numpy as np
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms 
import cv2
import json
from scipy.signal import gaussian

class AllInOneData(Dataset):
    def __init__(self, root_dir, set='train', required_types=[], transforms=None):
        self.types = required_types    
        self.root_dir = root_dir
        self.set_name = set
        self.transforms = transforms
        self.image_ids = self.getImgNames()
        self.loadClasses()

    def loadClasses(self):
        with open(os.path.join(self.root_dir,'config.json'),'r') as f:
            config = json.load(f)
        self.skin = config['skin']
        self.race  = config['race']
        self.gender = config['gender']
        self.emotion = config['emotions']


    def getImgNames(self):
        names = os.listdir(os.path.join(self.root_dir,self.set_name))
        names = [ x.split('.jpg')[0] for x in names if x.endswith('.jpg')]
        return names

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        
        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def load_image(self, image_index):
        #print(self.image_ids[image_index])
        path = os.path.join(self.root_dir,self.set_name, self.image_ids[image_index]+'.jpg')
        img = cv2.imread(path)
        self.img_h,self.img_w = img.shape[0:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def prepare_bbox(self,annot):
        bbox = np.array(annot,dtype=np.float16)
        if bbox.shape[0]==0:
            return np.array([-1,-1,-1,-1],dtype=np.float16)
        else:
            bbox[0:4:2] = bbox[0:4:2]#/self.img_w
            bbox[1:5:2] = bbox[1:5:2]#/self.img_h
            #np.clip(bbox,0,1,bbox)
        return bbox
    
    def prepare_cls(self,annot,key):
        prop = annot[key]
        if prop == [] or prop is None:
            prop = getattr(self,key).index("unknown")
        else:
            prop = getattr(self,key).index(str(annot[key]))
        return prop
    
    def prepare_landmarks(self,annot,key):
        if key == 'pose':
            out = np.ones((18,2))*-1
        elif key == 'face_landmarks':
            out = np.ones((70,2))*-1

        if annot[key] is not None and len(annot[key])!=0:
            for p in range(len(annot[key])):
                if type(annot[key][p]) is int or len(annot[key][p])==0:
                    continue
                out[p,:] = annot[key][p][:2]
        return out

        

    
    def prepare_age(self,annot):
        try:
            age = np.array(annot['age'],dtype=np.int8)    
            return age
        except TypeError:
            return -1


    def load_annotations(self,image_idx):
        #print(self.image_ids[image_idx])
        path = os.path.join(self.root_dir,self.set_name,self.image_ids[image_idx]+'_meta.json')
        with open(path,'r') as f:
            annot = json.load(f)
        for idx in annot.keys():
            annot[idx]['person_bbox'] = self.prepare_bbox(annot[idx]['person_bbox'])
            annot[idx]['face_bbox'] = self.prepare_bbox(annot[idx]['face_bbox'])
            annot[idx]['age'] = self.prepare_age(annot[idx]) 
            annot[idx]['gender'] = self.prepare_cls(annot[idx],'gender')
            annot[idx]['race'] = self.prepare_cls(annot[idx],'race')
            annot[idx]['skin'] = self.prepare_cls(annot[idx],'skin')
            annot[idx]['emotion'] = self.prepare_cls(annot[idx],'emotion')
            annot[idx]['pose'] = self.prepare_landmarks(annot[idx],'pose')
            annot[idx]['face_landmarks'] = self.prepare_landmarks(annot[idx],'face_landmarks')
        #print({a:annot[a].shape for a in annot.keys()})
        return annot
        
    
    
def ToTorch(annots,mbp,key):
    nam = [s[f'id{i+1}'][key] for s in annots for i in range(mbp)]
    tens = torch.from_numpy(np.stack(nam,axis=0))
    return tens

def collater(data):
    #### TO DO:  Now equalize the number of entries for each key for each person!!!!!
    imgs = [s['img'] for s in data]
    imgs = torch.from_numpy(np.stack(imgs, axis=0))
    imgs = imgs.permute(0, 3, 1, 2)

    annots = [s['annot'] for s in data]
    # Maximum persons on one image in batch
    mbp = max([len(list(i.keys())) for i in annots])
    #equlize number of persons on all the images and fill with dummies
    for annot in annots:   
        for k in range(mbp): 
            if f'id{k+1}' not in annot.keys():
                annot[f'id{k+1}'] = {}
                annot[f'id{k+1}']['age'] = np.array(-1).astype(np.int8)
                annot[f'id{k+1}']['gender'] = -1 
                annot[f'id{k+1}']['emotion'] = -1
                annot[f'id{k+1}']['race'] = -1
                annot[f'id{k+1}']['skin'] = -1
                annot[f'id{k+1}']['person_bbox'] = np.array([-1,-1,-1,-1]) 
                annot[f'id{k+1}']['face_bbox'] = np.array([-1,-1,-1,-1])
                annot[f'id{k+1}']['pose'] = np.ones((18,2))*-1
                annot[f'id{k+1}']['face_landmarks'] = np.ones((70,2))*-1

    age = ToTorch(annots,mbp,'age')
    skin = ToTorch(annots,mbp,'skin')
    race = ToTorch(annots,mbp,'race')
    emotion = ToTorch(annots,mbp,'emotion')
    gender = ToTorch(annots,mbp,'gender')

    person_bbox = ToTorch(annots,mbp,'person_bbox')
    face_bbox = ToTorch(annots,mbp,'face_bbox')
    pose = ToTorch(annots,mbp,'pose')
    face_landmarks = ToTorch(annots,mbp,'face_landmarks')

    return {'img': imgs, 
            'person_bbox': person_bbox.view((-1,mbp,4)),
            'face_bbox': face_bbox.view((-1,mbp,4)),
            'pose': pose.view((-1,mbp,18,2)),
            'face_landmarks': face_landmarks.view((-1,mbp,70,2)),
            'gender': gender.view((-1,mbp)),
            'age': age.view((-1,mbp)),
            'race': race.view((-1,mbp)),
            'emotion': emotion.view((-1,mbp)),
            'skin': skin.view((-1,mbp))}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image
        for idx in annots.keys():
            annots[idx]['person_bbox'] *= scale
            annots[idx]['face_bbox'] *= scale
            annots[idx]['pose'] *= scale
            annots[idx]['face_landmarks'] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': annots}



class Normalizer(object):

    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}


    


if __name__=='__main__':
    idx=0
    ds = AllInOneData('./datasets/Train2021',set='train',transforms = transforms.Compose([Normalizer(),Resizer()]))
    loader = DataLoader(ds,batch_size=3,shuffle=True,collate_fn=collater)
    batch = next(iter(loader))
    img = batch['img'][idx].permute(1,2,0)
    img = img.numpy()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    print(img.shape)
    gender = batch['gender']
    person_box = batch['person_bbox']
    print(person_box.shape)
    print(gender.shape)
    face = batch['face_bbox'][idx].numpy().astype(np.int64)
    landmarks = batch['face_landmarks'][idx].numpy().astype(np.int64)
    age = batch['age'][idx].numpy()
    print(age.shape)
    print(landmarks.shape)
    lm_mask = np.zeros((face.shape[0],1,256,256))
    lm = landmarks//2
    lm_mask[:,:,lm[:,:,1],lm[:,:,0]] = 1
    k_size=7
    krnl = gaussian(k_size,2).reshape(k_size,1)
    krnl = np.outer(krnl,krnl)*255
    #print(krnl)
    krnl = torch.from_numpy(krnl).view(1,1,k_size,k_size)
    # Gausian kernel for heatmap generation!!!
    lm_mask = torch.nn.functional.conv2d(torch.from_numpy(lm_mask).long(),krnl.long(),padding=(k_size-1)//2)
    lm_mask = lm_mask.numpy()
    lm_mask = lm_mask/np.max(lm_mask)
    
    print(face.shape)
    for p in range(face.shape[0]):        
        if face[p][0]!=-1:
            cv2.rectangle(img,(face[p][0],face[p][1]),(face[p][2],face[p][3]),(0,0,255),2)
        if age[p]!=0:
            cv2.putText(img,str(age[p]),(face[p][0],face[p][1]+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
        for i in range(0,landmarks[p].shape[0]):
            cv2.circle(img,(landmarks[p][i][0],landmarks[p][i][1]),3,(150,0,255),cv2.FILLED)
    while True:
        cv2.imshow('test',img)
        cv2.imshow('lm_mask',cv2.applyColorMap((255*lm_mask[0,0]).astype(np.uint8),cv2.COLORMAP_JET))
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

    
    