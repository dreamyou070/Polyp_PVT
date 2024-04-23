import torch
import argparse
from lib.pvt import PolypPVT
from torch import nn
def main(args) :

    print(f' step 1. make model')
    model = PolypPVT()
    #pretrained_pth_path = './model_pth/PolypPVT.pth'
    #model.load_state_dict(torch.load(pretrained_pth_path))
    pvt_encoder = model.backbone # pvtv2_b2 model

    pvt_encoder.

    """
    print(f' step 2. encoder')
    #pvt_encoder.cuda()
    pvt_encoder.eval()

    print(f' step 3. check encoder output')
    input_img = torch.randn(1, 3, 352, 352)#.cuda()
    encoder_output = pvt_encoder(input_img)



    x1 = encoder_output[0].permute(0, 2, 3, 1)
    x2 = encoder_output[1].permute(0, 2, 3, 1)
    x3 = encoder_output[2].permute(0, 2, 3, 1)
    x4 = encoder_output[3].permute(0, 2, 3, 1) # much deep feature !!
    # b,r,r,d -> b,p,d
    x1 = x1.reshape(1, -1, 64)
    x2 = x2.reshape(1, -1, 128)
    x3 = x3.reshape(1, -1, 320)
    x4 = x4.reshape(1, -1, 512)

    layer_1 = nn.Linear(64, 768)
    layer_2 = nn.Linear(128, 768)
    layer_3 = nn.Linear(320, 768)
    layer_4 = nn.Linear(512, 768)
    print(f'x1 = {x1.shape}')
    print(f'x2 = {x2.shape}')
    print(f'x3 = {x3.shape}')
    print(f'x4 = {x4.shape}')
    """




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int,
                        default=352, help='testing size')
    #parser.add_argument('--base_path', type=str,
    #                    default=r'/home/dreamyou070/MyData/anomaly_detection/medical/leader_polyp/Pranet/test')
    parser.add_argument('--save_base', type=str, default='./result_sy')
    parser.add_argument('--pth_path', type=str, default='./model_pth/PolypPVT.pth')
    args = parser.parse_args()
    main(args)
