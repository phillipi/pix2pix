from pdb import set_trace as st
import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='../dataset/50kshoes_edges')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='../dataset/50kshoes_jpg')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='../dataset/test_AB')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images',type=int, default=1000000)
parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) to (0001_AB)',action='store_true')
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg,  getattr(args, arg))


img_list = os.listdir(args.fold_A)
if args.use_AB: 
    img_list = [img_path for img_path in img_list if '_A.' in img_path]

num_imgs = min(args.num_imgs, len(img_list))
print('use %d/%d images' % (num_imgs, len(img_list)))
if not os.path.isdir(args.fold_AB):
    os.makedirs(args.fold_AB)
print('number of images = %d' % num_imgs)
for n in range(num_imgs):
    name_A = img_list[n]
    path_A = os.path.join(args.fold_A, name_A)
    if args.use_AB:
        name_B = name_A.replace('_A.', '_B.')
    else:
        name_B = name_A
    path_B = os.path.join(args.fold_B, name_B)
    if os.path.isfile(path_A) and os.path.isfile(path_B):
        name_AB = name_A
        if args.use_AB:
            name_AB = name_AB.replace('_A.', '.') # remove _A
        path_AB = os.path.join(args.fold_AB, name_AB)
        im_A = cv2.imread(path_A, cv2.CV_LOAD_IMAGE_COLOR)
        im_B = cv2.imread(path_B, cv2.CV_LOAD_IMAGE_COLOR)
        im_AB = np.concatenate([im_A, im_B], 1)
        cv2.imwrite(path_AB, im_AB)

