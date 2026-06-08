import os, sys, time, warnings, glob
import os.path as osp
import argparse
from types import SimpleNamespace

import numpy as np
from utils import paired_transforms_tv04 as p_tr
from PIL import Image
from skimage.io import imsave
from skimage.util import img_as_ubyte
import torch
from models.get_model import get_arch
from utils.model_saving_loading import load_model

import predict_one_image_av as P

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='experiments/big_wnet_drive_av')
parser.add_argument('--im_dir', required=True)
parser.add_argument('--result_path', required=True)
parser.add_argument('--tta', default='from_probs')
parser.add_argument('--im_size', default='512')
parser.add_argument('--device', default='cpu')
args = parser.parse_args()

# create_pred() in predict_one_image_av reads a module-global `args` for the device
P.args = args

device = torch.device(args.device)
tg_size = tuple(int(x) for x in args.im_size.split(','))
if len(tg_size) == 1:
    tg_size = (tg_size[0], tg_size[0])

os.makedirs(args.result_path, exist_ok=True)

print('* Instantiating model = big_wnet')
model = get_arch('big_wnet', n_classes=4).to(device)
model.mode = 'eval'
model, stats = load_model(model, args.model_path, device)
model.eval()

ims = sorted(glob.glob(osp.join(args.im_dir, '*.png')))
print(f'* Found {len(ims)} images')

t0 = time.perf_counter()
for i, im_path in enumerate(ims, 1):
    im_name = osp.basename(im_path)
    stem = im_name.rsplit('.', 1)[0]
    out_seg = osp.join(args.result_path, stem + '_seg.png')
    out_bin = osp.join(args.result_path, stem + '_bin_seg.png')
    if osp.exists(out_seg) and osp.exists(out_bin):
        print(f'[{i}/{len(ims)}] skip (exists) {im_name}', flush=True)
        continue
    try:
        img = Image.open(im_path).convert('RGB')
        mask = np.array(P.get_fov(img)).astype(bool)
        img_c, coords_crop = P.crop_to_fov(img, mask)
        original_sz = img_c.size[1], img_c.size[0]
        tr = p_tr.Compose([p_tr.Resize(tg_size), p_tr.ToTensor()])
        im_tens = tr(img_c)
        full_pred, full_pred_bin = P.create_pred(
            model, im_tens, mask, coords_crop, original_sz, tta=args.tta)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imsave(out_seg, img_as_ubyte(full_pred))
            imsave(out_bin, img_as_ubyte(full_pred_bin))
        print(f'[{i}/{len(ims)}] done {im_name}', flush=True)
    except Exception as e:
        print(f'[{i}/{len(ims)}] FAILED {im_name}: {e}', flush=True)

print('All done, total time = {:.1f} secs'.format(time.perf_counter() - t0))
