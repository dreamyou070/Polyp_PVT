from lib.pvt import PolypPVT
import torch

# call main model
model = PolypPVT().cuda()

# [2] cim module
backbone_model = model.backbone

# [3] input
input = torch.randn(1,3,256,256)
