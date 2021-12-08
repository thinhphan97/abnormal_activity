from models import ViT,ViViT
import torch
import numpy as np

# model = ViT(
#     image_size = 256,
#     patch_size = 32,
#     num_classes = 1000,
#     dim = 1024,
#     depth = 6,
#     heads = 16,
#     mlp_dim = 2048,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )
# parameters = filter(lambda p: p.requires_grad, model.parameters())
# parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
# print('Trainable Parameters: %.3fM' % parameters)

# img = torch.randn(1, 3, 256, 256)

# preds = model(img) # (1, 1000)

# print(preds.shape)

img = torch.ones([1, 16, 3, 224, 224])

model = ViViT(224, 16, 2, 16)
parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters: %.3fM' % parameters)

import time
t1 = time.time()
out = model(img)

print("Shape of out :", out.shape) 

print("Time :", time.time() - t1)
