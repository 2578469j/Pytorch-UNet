from unet import UNet, compute_local_variance
import numpy as np
import torch

#model = UNet(3, 3, bilinear=False)

inp = torch.ones((5,3,600,600))

res1 = compute_local_variance(inp, 3)
print(res1)
#res = model.forward(inp)