from torch import nn
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # aveage pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False) # res preserve,
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # sum of averaging pooling and max pooling
        # [1] avg pool out mean the average value
        y = self.avg_pool(x)
        print(f'y = {y.shape}')
        avg_out = self.fc2(self.relu1(self.fc1(y)))
        print(f'avg_out = {avg_out.shape}')
        # [2] max pool out mean the maximum value
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # [3] final is sum of two values
        out = avg_out + max_out # 1, channel, res, res

        # non linear mapping
        return self.sigmoid(out)
import torch
# avg_pool = spatial dimension averaging
# max_pool = spatial dimension max num
input = torch.randn(1,64,64,4)
model = ChannelAttention(in_planes = 4)
output = model(input)
print(f'output = {output.shape}')