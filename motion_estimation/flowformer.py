import sys
sys.path.append('core')

import numpy as np
import heapq as hq
import torch
import torch.nn.functional as F
from configs.submission import get_cfg

from core.FlowFormer import build_flowformer

from utils.utils import InputPadder

TRAIN_SIZE = [432, 960]

def build_model():
    print(f"building  model...")
    cfg = get_cfg()
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    return model

def prepare_image(image1, image2, image3, image4):
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
    image3 = torch.from_numpy(image3).permute(2, 0, 1).float()
    image4 = torch.from_numpy(image4).permute(2, 0, 1).float()

    return image1, image2, image3, image4

def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
    if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
        raise ValueError(
            f"Overlap should be less than size of patch (got {min_overlap}"
            f"for patch size {patch_size}).")
    if image_shape[0] == TRAIN_SIZE[0]:
        hs = list(range(0, image_shape[0], TRAIN_SIZE[0]))
    else:
        hs = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap))
    if image_shape[1] == TRAIN_SIZE[1]:
        ws = list(range(0, image_shape[1], TRAIN_SIZE[1]))
    else:
        ws = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap))

    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    return [(h, w) for h in hs for w in ws]

def compute_flow(model, image1, image2, weights=None):
    print(f"computing flow...")

    image_size = image1.shape[1:]
    image1, image2 = image1[None].cuda(), image2[None].cuda()

    hws = compute_grid_indices(image_size)
    if weights is None:     # no tile
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pre, _ = model(image1, image2)

        flow_pre = padder.unpad(flow_pre)
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()
    else:                   # tile
        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]    
            flow_pre, _ = model(image1_tile, image2_tile)
            padding = (w, image_size[1]-w-TRAIN_SIZE[1], h, image_size[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()
    return flow

def compute_motion(model, image_d1, image_d2, image_s1, image_s2):
    weights = None
    image_d1, image_d2, image_s1, image_s2 = prepare_image(image_d1, image_d2, image_s1, image_s2)
        
    # flow computation
    flow_d = compute_flow(model, image_d1, image_d2, weights)
    flow_s = compute_flow(model, image_s1, image_s2, weights)
    
    u = flow_d[:, :, 0] - flow_s[:, :, 0]
    v = flow_d[:, :, 1] - flow_s[:, :, 1]
    rad_diff = np.sqrt(np.square(u) + np.square(v))
    rad_d = np.sqrt(np.square(flow_d[:, :, 0]) + np.square(flow_d[:, :, 1]))
    rad = np.minimum(rad_diff, rad_d)
    return rad, flow_d, flow_s
    
def flow_to_gray(rad, min_flow=0.5, max_flow=2.5):
    """
    Transforms flow magnitude to grayscale image.
    """
    rad = np.clip(rad, min_flow, max_flow)
    p = (rad - min_flow) / (max_flow - min_flow) # probability
    
    return p

def flow_to_gray_norm(rad):
    """
    Transforms flow magnitude to grayscale image.
    """
    num = int(0.01 * rad.shape[0] * rad.shape[1]) 
    rad_ = rad.copy()
    max_flow= np.mean(hq.nlargest(num, rad_.flatten()))
    rad_[rad_<= 0.01] = 10
    min_flow = np.mean(hq.nsmallest(num, rad_.flatten()))

    p = 1 / (1 + np.exp(-(max_flow))) * (rad - min_flow) / (max_flow - min_flow) # probability
    p = np.clip(p, 0, 1)
    return p

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, rad_max=10, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        rad_max (float, optional): max flow magnitude for clipping. Defaults to 10.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    u = u / rad_max
    v = v / rad_max
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image