import cv2
import numpy as np   
from albumentations import pytorch
import albumentations as A 
from defaults import _C
import glob
import torch

totensor = pytorch.transforms.ToTensorV2() 
normalize = A.Normalize()

cfg = _C

images = glob.glob("image/video_166/*" )
images = sorted(images)
length = len(images)
data=[]
for i in range(0, length-(cfg.DATA.NUM_SLICES*cfg.DATA.STRIDE -2),cfg.DATA.STEP):
    image_slice = []
    for j in range(0,cfg.DATA.NUM_SLICES*cfg.DATA.STRIDE,cfg.DATA.STRIDE):       
        img = cv2.imread(images[i+j])
        img = cv2.resize(img,(224,224))
        # img_tensor = normalize(image=img)["image"]
        # img_tensor = totensor(image=img)["image"]
        image_slice.append(img)       
    # image_slice = torch.stack(image_slice)
    data.append(image_slice)
# data =  torch.stack(data)
data = np.array(data)
batch, _, _, _, _=data.shape
for i in range(batch):
    print("Mean data:", data[i].mean())
    print("Standard deviation:", data[i].std())
