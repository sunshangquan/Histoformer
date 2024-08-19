import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import util

from natsort import natsorted
from glob import glob
import sys
sys.path.append(os.path.join(os.getcwd(), ".."))
from basicsr.models.archs.histoformer_arch import Histoformer
from skimage import img_as_ubyte
from pdb import set_trace as stx
import time
parser = argparse.ArgumentParser(description='Image Deraining using Restormer')

parser.add_argument('--input_dir', default='./Datasets/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/deraining.pth', type=str, help='Path to weights')
parser.add_argument('--yaml_file', default='Options/Allweather_Histoformer.yml', type=str, help='Path to weights')

args = parser.parse_args()

####### Load yaml #######
yaml_file = args.yaml_file
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = Histoformer(**x['network_g'])

checkpoint = torch.load(args.weights)
'''
from thop import profile
flops, params = profile(model_restoration, inputs=(torch.randn(1, 3, 256,256), ))
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')
'''
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

factor = 8

result_dir  = os.path.join(args.result_dir)
os.makedirs(result_dir, exist_ok=True)
inp_dir = os.path.join(args.input_dir)
files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
with torch.no_grad():
    for file_ in tqdm(files):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(util.load_img(file_))/255.
        img = torch.from_numpy(img).permute(2,0,1)
        input_ = img.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        time1 = time.time()
        restored = model_restoration(input_)
        time2 = time.time()
        #print(time2-time1)

        # Unpad images to original dimensions
        restored = restored[:,:,:h,:w]

        restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        util.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'.png')), img_as_ubyte(restored))
