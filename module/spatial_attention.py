from torch import nn
import torch
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # spatial mean
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # max value
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

input = torch.randn(1,4,64,64)
mean = torch.mean(input, dim=1, keepdim = True) # 1,1,64,64
print(mean.shape)

# to add more