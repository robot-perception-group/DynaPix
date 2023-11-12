# Generate Estimated Motion Masks
# Last Modified: 11/05/2023

import os
import confuse
import argparse

import torch
import cv2 as cv
import heapq as hq
import numpy as np
import pickle as pkl
from PIL import Image
from pyquaternion import Quaternion

from Warper import Warper
from flowformer import build_model, compute_motion, flow_to_gray, flow_to_gray_norm

class MotionGT(object):
    def __init__(self, config):
        self.DEBUG = config['debug'].get()
        self.OUTPUT = config['output'].get()

        # initialize params
        camera_params = config['camera'].get()

        # intrinsic camera matrix
        self.width = camera_params[0]
        self.height = camera_params[1]
        self.fx = camera_params[2]
        self.fy = camera_params[3]
        self.cx = camera_params[4]
        self.cy = camera_params[5]
        
        self.K = np.array([[self.fx,  0, self.cx],
                           [ 0, self.fy, self.cy],
                           [ 0,  0,  1]])
        
        # initialize warper
        self.warper = Warper(self.K, self.width, self.height)
        
        self.path = config['path'].get()
        self.output_path = config['output_path'].get()
        
        with open(os.path.join(self.path, 'association.txt'),'r') as f:
            data = f.readlines()
            self.matches = []
            for d in data:
                if '#' in d: continue
                self.matches.append(d)

        with open(os.path.join(self.path, 'groundtruth.txt'),'r') as f:
            data = f.readlines()
            self.pose_gt = {}
            for d in data:
                if '#' in d: continue
                d = d.split('\n')[0].split(' ')
                self.pose_gt[float(d[0])] = [float(d[i]) for i in range(1, 8)]
    
    def get_movable(self, dynaImg, staticImg, depth, masks=None):
        img_diff = cv.absdiff(dynaImg, staticImg) # extract the difference between static and dynamic images
        # resize depth map
        depth = cv.resize(depth, (self.width, self.height), interpolation=cv.INTER_NEAREST)
        
        diff0 = np.max(img_diff, axis=2).astype(float)
        diff1 = np.mean(img_diff, axis=2).astype(float)
        
        # threshold difference
        diff0 = np.clip(diff0, a_min=15.0, a_max=35.0)
        mask = (diff0 - 15.0) / 20.0
        mask[depth >= 10] = 0.0
        
        # normalize difference
        num = int(self.width * self.height) 
        diff1_ = diff1.copy()
        max_diff = np.mean(hq.nlargest(int(0.05*num), diff1_.flatten()))
        diff1_[diff1_ <= 5] = 100
        min_diff = np.mean(hq.nsmallest(int(0.1*num), diff1_.flatten()))

        mask_norm = (diff1 - min_diff) / (max_diff - min_diff + 1e-6)
        mask_norm = np.clip(mask_norm, 0, 1)
        mask_norm[depth >= 10] = 0.0

        mask_lambda = 0.5 + 1 / (np.exp(0.04 * max_diff) + 1)
        weighted_mask = mask * mask_lambda + mask_norm * (1 - mask_lambda)
        return weighted_mask, mask, mask_norm

    def search_pose(self, img_t):
        if img_t in list(self.pose_gt.keys()):
            pose = self.pose_gt[img_t]
        else:
            delta_t = 1e6
            for k in self.pose_gt.keys():
                delta_t_ = abs(img_t - k)
                if  delta_t_ < delta_t:
                    delta_t = delta_t_
                    pose_t = k
                    pose = self.pose_gt[pose_t]

        T = Quaternion(w=pose[6], x=pose[3], y=pose[4], z=pose[5]).transformation_matrix
        T[:3,3] = [pose[0],pose[1],pose[2]]
        
        return T
    

    @staticmethod
    def pose2T(poseFile):
        '''
        Convert pose to transformation matrix
        '''
        pose = np.load(poseFile, allow_pickle = True).item()['pose']
        T = pose.T
        T[:3,3] = T[:3,3] * 0.01
        T[:3,1:3] = -T[:3,1:3]
        return T
    
    @staticmethod
    def detect_occlusion(rgb, depth, depth_thr): # todo add segmentation perhaps
        rgb_mask = np.zeros(rgb.shape, dtype=np.uint8)
        depth_mask = np.zeros(rgb.shape, dtype=np.uint8)
        
        rgb_mask[np.where((rgb <= [15,15,15]).all(axis=2))] = [255,255,255]
        depth_mask[depth < depth_thr] = [255,255,255]
        
        # calculate the percentage of the rgb / depth are occluded
        perc_rgb = (np.count_nonzero(rgb_mask) / (3 * rgb.shape[0] * rgb.shape[1])) * 100
        perc_depth = (np.count_nonzero(depth_mask) / (3 * rgb.shape[0] * rgb.shape[1])) * 100
        
        return perc_rgb, perc_depth
        

    # main function
    def generate_masks(self, model):
        with torch.no_grad():
            for i in range(1, len(self.matches)+1):
                if i < 2 or i > len(self.matches) - 1:
                    pm = np.ones((self.height, self.width), dtype=np.uint16)
                    
                    if self.OUTPUT:
                        Image.fromarray((pm*1000)).save(os.path.join(self.output_path, f'{i}.png'))
                else:
                    img_t = float(self.matches[i-1].split(' ')[0])
                    rgb_fn = self.matches[i-1].split('\n')[0].split(' ')[1]
                    depth_fn = self.matches[i-1].split('\n')[0].split(' ')[3]
                    print(f'Generating mask for image {rgb_fn} ...')

                    # load poses
                    T_cur = self.search_pose(img_t)
                    img_s = cv.imread(os.path.join(self.path, rgb_fn.replace('rgb','background')), cv.IMREAD_UNCHANGED)
                    img_d = cv.imread(os.path.join(self.path, rgb_fn),cv.IMREAD_UNCHANGED)
                    img_s = cv.resize(img_s, (self.width, self.height), interpolation=cv.INTER_NEAREST)
                    img_d = cv.resize(img_d, (self.width, self.height), interpolation=cv.INTER_NEAREST)
                    
                    depth = cv.imread(os.path.join(self.path, depth_fn), cv.IMREAD_ANYDEPTH)/5000.
                    
                    # load masks
                    weighted_mask, mask, mask_norm = self.get_movable(img_d, img_s, depth)
                    

                    # load images from previous frame
                    img_t_prev = float(self.matches[i-2].split(' ')[0])
                    T_prev = self.search_pose(img_t_prev)
                    
                    rgb_fn = self.matches[i-2].split('\n')[0].split(' ')[1]
                    depth_fn = self.matches[i-2].split('\n')[0].split(' ')[3]
                    img_d_prev = cv.imread(os.path.join(self.path, rgb_fn), cv.IMREAD_UNCHANGED)
                    img_s_prev = cv.imread(os.path.join(self.path, rgb_fn.replace('rgb','background')), cv.IMREAD_UNCHANGED)
                    depth_prev = cv.imread(os.path.join(self.path, depth_fn), cv.IMREAD_ANYDEPTH)/5000.

                    # warp previous images
                    img_d_prev = self.warper.forward_warp(img_d_prev, T_prev, T_cur, 'splatting', depth_prev)
                    img_s_prev = self.warper.forward_warp(img_s_prev, T_prev, T_cur, 'splatting', depth_prev)
                    
                    
                    # load images from next frame
                    img_t_next = float(self.matches[i].split(' ')[0])
                    T_next = self.search_pose(img_t_next)

                    rgb_fn = self.matches[i].split('\n')[0].split(' ')[1]
                    depth_fn = self.matches[i].split('\n')[0].split(' ')[3]
                    img_d_next = cv.imread(os.path.join(self.path, rgb_fn), cv.IMREAD_UNCHANGED)
                    img_s_next = cv.imread(os.path.join(self.path, rgb_fn.replace('rgb','background')), cv.IMREAD_UNCHANGED)
                    depth_next = cv.imread((os.path.join(self.path, depth_fn)), cv.IMREAD_ANYDEPTH)/5000.
                    
                    # warp next images
                    img_s_next = self.warper.forward_warp(img_s_next, T_next, T_cur, 'splatting', depth_next)
                    img_d_next = self.warper.forward_warp(img_d_next, T_next, T_cur, 'splatting', depth_next)
                    
                    # moving estimation
                    rad_prev, flow_d_prev, flow_s_prev = compute_motion(model, img_d, img_d_prev, img_s, img_s_prev)
                    rad_next, flow_d_next, flow_s_next = compute_motion(model, img_d, img_d_next, img_s, img_s_next)
                    rad = 0.5 * (rad_prev + rad_next)
                    
                    # motion probability maps
                    pm = weighted_mask * flow_to_gray(rad, 1, 4) # similar performance of flow_to_gray_norm
                    
                    if self.OUTPUT:
                        Image.fromarray(((1 - pm)*1000).astype(np.uint16)).save(os.path.join(self.output_path, f'{i}.png'))
                    
                    if self.DEBUG:
                        img_movable1 = (mask*255).astype(np.uint8)
                        img_movable2 = (mask_norm*255).astype(np.uint8)
                        img_movable = (weighted_mask*255).astype(np.uint8)

                        img_moving = (flow_to_gray(rad) *255).astype(np.uint8)
                        # img_moving_ = (flow_to_gray_norm(rad) *255).astype(np.uint8)
                        img_motion = (pm*255).astype(np.uint8)

                        cv.imshow('img_movable1', img_movable1)
                        cv.imshow('img_movable2', img_movable2)
                        cv.imshow('img_movable', img_movable)
                        cv.imshow('img_moving', img_moving)
                        # cv.imshow('img_moving_norm', img_moving_)
                        cv.imshow('img_motion', img_motion)
                        cv.waitKey(0)

if __name__ == '__main__':
    # Define parser argumeenennts
    parser = argparse.ArgumentParser(description="pixel-wise moving probability")
    parser.add_argument("--config", type=str, default="configs/TUM.yaml", help="Path to Config File")
    parser.add_argument("--debug", action="store_true", help="Debug Mode")
    parser.add_argument("--output", action="store_true", help="Saving Motion Probability")
    args, _ = parser.parse_known_args()
    
    # load configuration file
    config = confuse.Configuration("pixel-wise moving probability", __name__)
    config.set_file(args.config)
    config.set_args(args)

    output_path = config['output_path'].get()
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model = build_model()
    motion = MotionGT(config)
    
    # Generate probabilistic masks
    motion.generate_masks(model)
