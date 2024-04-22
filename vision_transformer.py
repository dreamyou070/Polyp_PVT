import torch
import argparse
from lib.pvt import PolypPVT
def main(args) :

    print(f' step 1. make model')
    model = PolypPVT()
    #pretrained_pth_path = './model_pth/PolypPVT.pth'
    #model.load_state_dict(torch.load(pretrained_pth_path))
    pvt_encoder = model.backbone # pvtv2_b2 model

    print(f' step 2. encoder')
    #pvt_encoder.cuda()
    pvt_encoder.eval()

    print(f' step 3. check encoder output')
    input_img = torch.randn(1, 3, 352, 352)#.cuda()
    encoder_output = pvt_encoder(input_img)



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
