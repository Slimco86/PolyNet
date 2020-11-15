import os
from tqdm import tqdm
import shutil
import numpy as np

base_path = './Data'#os.path.join(os.curdir,'Data')
base_path_copy = './datasets'
total_count=0
split = 0.8

folders = os.listdir(base_path)
folders = [x for x in folders if os.path.isdir(os.path.join(base_path,x))]
for folder in tqdm(folders):
    imgs = os.listdir(os.path.join(base_path,folder))
    imgs = [x for x in imgs if x.endswith('.jpg')]
    img_cnt = len(imgs)
    train = int(img_cnt*split)
    valid = int(img_cnt*(1-split))
    i = 0
    while i < train:
        i+=1
        ind = np.random.randint(len(imgs))
        img_name = imgs.pop(ind)
        anot_name = img_name.split('.jpg')[0]+'_meta.json'
        shutil.copy(os.path.join(base_path,folder,img_name),os.path.join(base_path_copy,'Train2020/train',img_name))
        shutil.copy(os.path.join(base_path,folder,anot_name),os.path.join(base_path_copy,'Train2020/train',anot_name))

    for img in imgs:
        img_name = img
        anot_name = img_name.split('.jpg')[0]+'_meta.json'
        shutil.copy(os.path.join(base_path,folder,img_name),os.path.join(base_path_copy,'Train2020/validation',img_name))
        shutil.copy(os.path.join(base_path,folder,anot_name),os.path.join(base_path_copy,'Train2020/validation',anot_name))    

    total_count+=img_cnt

print(total_count)



