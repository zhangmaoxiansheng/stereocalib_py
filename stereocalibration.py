import numpy as np
from calibration import Calibrator, Rectify
import argparse
import os
import cv2
'''
opencv stereo camera(or single camera) calibration
usage:
    args pattern should be a tuple
    1)single camera intrinsic: --K the intrinsic matrix save path --img_path single camera's imgs
    2)stereo camera intrinsic: --K and --img_path for two camras
    3)stereo camera intrinsic and extrinsic: with intrinsic-> --K the path to intrinsic only use stereocalibration
                                             without intrinsic->--K: intrinsic save path use cameracalibration and stereocalibraion
    4)rectify 
tips: if you use different imgs for intrinsic calbiration and extrinsic calibraion, setImgPath again! 
warning: matlab often works bettern than opencv                                     
'''
parser = argparse.ArgumentParser(description='stereocalibration')
parser.add_argument('--K', nargs = '+', default=['../data/6mm_stereo/cam130.xml','../data/6mm_stereo/cam132.xml'] ,
                    help='cam intrinsics path')
parser.add_argument('--img_path', nargs = '+', default=['../data/6mm_stereo/cam130', '../data/6mm_stereo/cam132'],
                    help='imgs path')
parser.add_argument('--pattern', type = int, nargs = '+', default= (7,6),
                    help='pattern size,tuple')
parser.add_argument('--sq_size', type = float, default= 300,
                    help='square size, mm')
parser.add_argument('--calibrate_res', default='./calib.xml',
                    help='save stereo calibraion xml file')
parser.add_argument('--rectify_path', nargs= '+',default=['../result/130','../result/132'],
                    help='imgs need to be rectified, left images and right images')
parser.add_argument('--save_rectify', default='../data/cam3/res',
                    help='rectify res')
parser.add_argument('--rectify_only', action="store_true",
                    help='rectify res')
args = parser.parse_args()

if not args.rectify_only:
    calibrator = Calibrator(args.K, args.img_path, args.pattern, args.sq_size, args.calibrate_res)
    calibrator.findCorners()
    calibrator.stereoCalibration()
    print('finish calibration!')
if os.path.exists(args.calibrate_res):
    rec = Rectify(args.calibrate_res)
    rectify_path_left = os.path.join(args.rectify_path[0])
    rectify_path_right = os.path.join(args.rectify_path[1])

if os.path.exists(rectify_path_left) and os.path.exists(rectify_path_right):
    left_imgs = os.listdir(rectify_path_left)
    right_imgs = os.listdir(rectify_path_right)
    if len(left_imgs) == len(right_imgs):
        save_rectify = args.save_rectify
        if not os.path.exists(save_rectify):
            os.mkdir(save_rectify)
        left_res_f = os.path.join(save_rectify,'left')
        right_res_f = os.path.join(save_rectify,'right')
        if not os.path.exists(left_res_f):
            os.mkdir(left_res_f)
        if not os.path.exists(right_res_f):
            os.mkdir(right_res_f)

        left_imgs.sort()
        right_imgs.sort()
        
        for left_file,right_file in zip(left_imgs,right_imgs):
            left_file_ = os.path.join(rectify_path_left, left_file)
            right_file_ = os.path.join(rectify_path_right, right_file)

            left_img = cv2.imread(left_file_)
            right_img = cv2.imread(right_file_)
            left_rec, right_rec = rec(left_img, right_img)
            cv2.imwrite(os.path.join(left_res_f, left_file), left_rec)
            cv2.imwrite(os.path.join(right_res_f, right_file), right_rec)
            
    else:
        print('the number of imgs in two folder is not equal')

else:
    print("the rectify path should contains left and right folder")

    
