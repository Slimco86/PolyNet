import os
from tqdm import tqdm
import shutil
import numpy as np
import json

base_path='./datasets/Train2020/validation'

files = os.listdir(base_path)
annots = filter(lambda x: x.endswith('.json'),files)
valid_keys = ["skin","pose","race","emotion","person_bbox","face_bbox","face_landmarks","age","gender"]
for file in annots:
    with open (os.path.join(base_path,file)) as fl:
        f = json.load(fl)
    
    for key in valid_keys:
        if key not in f.keys():
            print(file,key)
