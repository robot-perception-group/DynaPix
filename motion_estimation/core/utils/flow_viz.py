# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import numpy as np

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


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
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

def flow_uv_to_gray(rad, min_flow=1, max_flow=5):
    """
    Transforms flow magnitude to grayscale image.
    """
    rad[np.where(rad<=min_flow)] = min_flow
    rad[np.where(rad>=max_flow)] = max_flow
    rad = (rad - min_flow) / (max_flow-min_flow)
    
    # flow_image = rad.astype(np.uint8)
    return rad

def flow_uv_to_gray_norm(rad):
    """
    Transforms flow magnitude to grayscale image.
    """
    min_flow = np.min(rad)
    max_flow = np.max(rad)
    
    rad = np.abs(rad - min_flow) / (max_flow-min_flow)
    
    #flow_image = rad.astype(np.uint8)
    return rad


def flow_to_image_mask(flow_uv, flow_static, mask, clip_flow=None, convert_to_bgr=False, max_flow=None):
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0] - flow_static[:,:,0]
    v = flow_uv[:,:,1] - flow_static[:,:,1]

    if max_flow is None:
        rad = np.sqrt(np.square(u) + np.square(v))
        #rad_max = np.max(rad)
    else:
        rad_max = max_flow
    u_bg = np.mean(u[mask])
    v_bg = np.mean(v[mask])
    if np.isnan(u_bg) or np.isnan(v_bg):
        u = u / (rad_max + epsilon)
        v = v / (rad_max + epsilon)
    else:
        u = (u - u_bg) / (rad_max + epsilon)
        v = (v - v_bg) / (rad_max + epsilon)
    u[mask] = 0
    v[mask] = 0
    del mask, flow_uv
    return flow_uv_to_colors(u, v, convert_to_bgr)

def flow_to_image_static(flow_uv, flow_uv2, flow_uv_static, flow_uv_static2, mask, mask_norm, clip_flow=None, max_flow=None):
    # flow difference
    u = flow_uv[:,:,0] - flow_uv_static[:,:,0]
    v = flow_uv[:,:,1] - flow_uv_static[:,:,1]
    
    # previous flow difference
    u_prev = flow_uv2[:,:,0] - flow_uv_static2[:,:,0]
    v_prev = flow_uv2[:,:,1] - flow_uv_static2[:,:,1]

    # flow magnitude
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_ = np.sqrt(np.square(flow_uv[:,:,0]) + np.square(flow_uv[:,:,1]))
    
    # previous flow magnitude
    rad_prev= np.sqrt(np.square(u_prev) + np.square(v_prev))
    rad_prev_= np.sqrt(np.square(flow_uv2[:,:,0]) + np.square(flow_uv2[:,:,1]))

    rad = np.minimum(rad,rad_)
    rad_prev = np.minimum(rad_prev,rad_prev_)
    
    rad_mask = 0.5 * (rad + rad_prev) * mask
    rad_mask_norm = 0.5 * (rad + rad_prev) * mask_norm
    #import ipdb; ipdb.set_trace()
    p_mask1 = flow_uv_to_gray(rad_mask)
    p_mask2 = flow_uv_to_gray_norm(rad_mask)
    p_mask3 = flow_uv_to_gray(rad_mask_norm)
    p_mask4 = flow_uv_to_gray_norm(rad_mask_norm)
    
    flow_image = 0.25 * (p_mask1 + p_mask2 + p_mask3 + p_mask4) * 255
    flow_image = flow_image.astype(np.uint8)
    img1 = (p_mask1*255).astype(np.uint8)
    img2 = (p_mask2*255).astype(np.uint8)
    img3 = (p_mask3*255).astype(np.uint8)
    img4 = (p_mask4*255).astype(np.uint8)
    
    return flow_image, img1, img2, img3, img4


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False, max_flow=None):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    if max_flow is None:
        rad = np.sqrt(np.square(u) + np.square(v))
        rad_max = np.max(rad)
        print(rad_max)
    else:
        rad_max = max_flow
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_gray(rad, min_flow = 0, max_flow=10), flow_uv_to_colors(u, v, convert_to_bgr)