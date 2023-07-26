import os
import cv2
import argparse
from utils import calculate_psnr, calculate_ssim


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='./Test1/result/')
    parser.add_argument('--target_dir', type=str, default='./Test1/target/')
    args = parser.parse_args()

    path_result = args.result_dir
    path_target = args.target_dir
    image_list = os.listdir(path_target)
    L = len(image_list)
    psnr_total, ssim_total = 0, 0
    for i in range(L):

        image_in = cv2.imread(path_result + str(image_list[i]), cv2.IMREAD_COLOR)
        image_tar = cv2.imread(path_target + str(image_list[i]), cv2.IMREAD_COLOR)

        ps = calculate_psnr(image_in, image_tar, test_y_channel=True)
        ss = calculate_ssim(image_in, image_tar, test_y_channel=True)

        psnr_total += ps
        ssim_total += ss

        print(f'\r{i} / {L}, PSNR: {ps}, SSIM: {ss}', end='', flush=True)

    print(f'\n PSNR: {psnr_total / L}, SSIM: {ssim_total / L}')