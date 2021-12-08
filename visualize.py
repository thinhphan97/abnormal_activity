import seaborn
import matplotlib.pyplot as plt
from albumentations import pytorch
import albumentations as A
from models import ViViT
from defaults import _C
import glob
import cv2
import numpy as np
import torch
from torch.nn import Softmax
def draw(data,ax):
    seaborn.heatmap(data, square=True, vmin=0.0, vmax=1.0, 
                    cbar=False, annot=False, ax=ax)


cfg = _C
model = ViViT(cfg)
ckpt = torch.load('checkpoint/vivit_224.pth', "cpu")
model.load_state_dict(ckpt.pop('state_dict'))


images = glob.glob("image/video_166/*" )
images = sorted(images)
length = len(images)
# print(length)
totensor = pytorch.transforms.ToTensorV2() 
normalize = A.Normalize()
data=[]
for i in range(0, length-(cfg.DATA.NUM_SLICES*cfg.DATA.STRIDE -2),cfg.DATA.STEP):
    image_slice = []
    for j in range(0,cfg.DATA.NUM_SLICES*cfg.DATA.STRIDE,cfg.DATA.STRIDE):       
        img = cv2.imread(images[i+j])
        img = cv2.resize(img,(224,224))
        img_tensor = normalize(image=img)["image"]
        img_tensor = totensor(image=img_tensor)["image"]
        image_slice.append(img_tensor)       
    image_slice = torch.stack(image_slice)
    data.append(image_slice)
ckpt = torch.load('checkpoint/vivit_224.pth', "cpu")
model.load_state_dict(ckpt.pop('state_dict'))
# img =  np.array(image_slice)
image_slice = torch.stack(data)
# image_slice = image_slice.unsqueeze(0)
print(Softmax(dim = 1)(model(image_slice[19].unsqueeze(0))))

for layer in range(0, 4):
    fig, axs = plt.subplots(1,3, figsize=(17, 17))
    print("temporal_transformer ", layer+1)
    for h in range(3):
        draw(model.temporal_transformer.layers[layer][0].fn.attn[0,h].data.cpu(),ax=axs[h])
    plt.show()

# print(model.temporal_transformer.layers[0][0].fn.attn.data.cpu().shape)
for layer in range(0, 4):
    fig, axs = plt.subplots(1,3, figsize=(49, 49))
    print("space_transformer ", layer+1)
    for h in range(3):
        draw(model.space_transformer.layers[layer][0].fn.attn[0,h].data.cpu(),ax=axs[h])
    plt.show()