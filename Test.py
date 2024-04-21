import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.pvt import PolypPVT
from utils.dataloader import test_dataset
import cv2
from PIL import Image
def main(args) :

    print(f' step 1. make model')
    model = PolypPVT()
    model.load_state_dict(torch.load(args.pth_path))
    model.cuda()
    model.eval()

    print(f' step 2. check data path')
    for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:

        # [1] data_path here
        data_path = os.path.join(args.base_path, _data_name)

        # [2] save_path
        save_path = os.path.join(args.save_base, _data_name) # './result_map/PolypPVT/{}/'.format()
        os.makedirs(save_path, exist_ok=True)


        image_root = os.path.join(data_path, 'images')
        gt_root = os.path.join(data_path, 'masks')

        num1 = len(os.listdir(gt_root))

        # [3] make test dataset
        test_loader = test_dataset(image_root, gt_root, 352)

        for i in range(num1):

            # [4] load data
            # image
            # gt = pil image
            # name
            image, gt, name, rgb_image = test_loader.load_data()
            gt = np.asarray(gt, np.float32) # [384,384]
            gt /= (gt.max() + 1e-8)
            image = image.cuda()



            # [5] model forward
            P1, P2 = model(image)

            # [6] eval Dice
            res = F.upsample(P1 + P2, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            # [7] merging two image with alpha value
            # res = [500,574]
            # [0 ~ 1 ] value
            h,w = res.shape
            # expand rgb_image to [500,574,3]
            res = np.expand_dims(res, axis=2).repeat(3, axis=2)
            res_pil = Image.fromarray((res * 255).astype(np.uint8))


            rgb_image = rgb_image.resize((w,h))
            #rgb_np = np.array(rgb_image) / 255
            #print(f'rgb_np shape : {rgb_np.shape}')

            #res = cv2.addWeighted(rgb_np, 0.6, res, 0.4, 0) # res (bad black position white)
            #cv2.imwrite(os.path.join(save_path, f'{name}'), res * 255)

            merged_image = Image.blend(rgb_image, res_pil, 0.4)

            # [8] merging all image
            total_img = Image.new('RGB', (w*3, h))
            total_img.paste(rgb_image, (0,0))
            total_img.paste(res_pil, (w,0))
            total_img.paste(merged_image, (w*2,0))
            total_img.save(os.path.join(save_path, f'{name}'))
            
        print(_data_name, 'Finish!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int,
                        default=352, help='testing size')
    parser.add_argument('--base_path', type=str,
                        default=r'/home/dreamyou070/MyData/anomaly_detection/medical/leader_polyp/Pranet/test')
    parser.add_argument('--save_base', type=str, default='./result_sy')
    parser.add_argument('--pth_path', type=str, default='./model_pth/PolypPVT.pth')
    args = parser.parse_args()
    main(args)
