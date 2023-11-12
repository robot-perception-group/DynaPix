# Warps frame using pose info
# Last Modified: 11/05/2023

import numpy as np
import cv2 as cv

class Warper:
    def __init__(self, K, width, height):
        """
        Initializes the Warper Class

        Args:
        :param K (3, 3): camera intrinsic matrix
        :param width (int): width of the image frame
        :param height (int): height of the image frame
        """
        self.h = height
        self.w = width
        self.K= K
        return

    def forward_warp(self, frame, transformation1, transformation2, method='splatting', depth=None):
        """
        Given a frame1 and global transformations transformation1 and transformation2, warps frame1 to next view
        
        Args:
        :param frame: (h, w, 3) uint8 np array
        :param depth: (h, w) float np array
        :param transformation1: (4, 4) extrinsic transformation matrix of first view: [R, t; 0, 1]
        :param transformation2: (4, 4) extrinsic transformation matrix of second view: [R, t; 0, 1]
        """
        frame = cv.resize(frame, (self.w, self.h), interpolation=cv.INTER_NEAREST)
        
        # transformed coordinates
        if method == 'homo_transform':
            trans_coor = self.compute_transformed_points(transformation1, transformation2, method)
            
            trans_norm_coor = (trans_coor[:, :, :2, 0] / trans_coor[:, :, 2:3, 0]).astype('float32')
            warped_frame = cv.remap(frame, trans_norm_coor[:,:,0], trans_norm_coor[:,:,1], cv.INTER_NEAREST)
        elif method == 'remap':
            depth = cv.resize(depth, (self.w, self.h), interpolation=cv.INTER_NEAREST)
            assert depth.shape[0] == self.h
            assert depth.shape[1] == self.w
            trans_coor = self.compute_transformed_points(transformation1, transformation2, method, depth)
            
            trans_norm_coor = (trans_coor[:, :, :2, 0] / trans_coor[:, :, 2:3, 0]).astype('float32')
            warped_frame = cv.remap(frame, trans_norm_coor[:,:,0], trans_norm_coor[:,:,1], cv.INTER_NEAREST)
        elif method == 'splatting':
            depth = cv.resize(depth, (self.w, self.h), interpolation=cv.INTER_NEAREST)
            assert depth.shape[0] == self.h
            assert depth.shape[1] == self.w
            trans_coor = self.compute_transformed_points(transformation1, transformation2, method, depth)
            
            trans_norm_coor = (trans_coor[:, :, :2, 0] / trans_coor[:, :, 2:3, 0]).astype('float32')
            trans_depth = trans_coor[:, :, 2, 0]
            warped_frame, _ = self.bilinear_splatting(frame, trans_depth, trans_norm_coor)
        else:
            raise ValueError('Invalid method') 
        
        return warped_frame

    def compute_transformed_points(self, trans_mat1, trans_mat2, method, depth=None):
        """
        Computes transformed position for each pixel location
        """
        # Define Image Meshgrid
        x2d, y2d = np.meshgrid(np.arange(self.w), np.arange(self.h))
        ones_2d = np.ones(shape=(self.h, self.w))
        pos_vectors_homo = np.stack([x2d, y2d, ones_2d], axis=2)[:, :, :, None]
        
        if method == 'homo_transform':
            trans_mat = np.matmul(np.linalg.inv(trans_mat2), trans_mat1)
            R = trans_mat[:3, :3]
            T = np.zeros((3, 3))
            T[:, 2] = trans_mat[:3, 3]
            H = np.matmul(np.matmul(self.K, R+T), np.linalg.inv(self.K))[None, None]
            H = np.linalg.inv(H)
            
            trans_coor = np.matmul(H, pos_vectors_homo)
            
        elif method == 'remap':
            trans_mat = np.matmul(np.linalg.inv(trans_mat1), trans_mat2)
            depth = depth[:, :, None, None]
            
            H = np.matmul(np.matmul(self.K, trans_mat[:3, :3]), np.linalg.inv(self.K))[None, None]
            t = np.matmul(self.K, trans_mat[:3,3]).reshape(3,1)
            
            trans_coor = np.matmul(H, depth * pos_vectors_homo) + t
        
        elif method == 'splatting':
            trans_mat = np.matmul(np.linalg.inv(trans_mat2), trans_mat1)
            depth = depth[:, :, None, None]
            
            H = np.matmul(np.matmul(self.K, trans_mat[:3, :3]), np.linalg.inv(self.K))[None, None]
            t = np.matmul(self.K, trans_mat[:3,3]).reshape(3,1)
            
            trans_coor = np.matmul(H, depth * pos_vectors_homo) + t
        return trans_coor

    def bilinear_splatting(self, frame, depth, trans_coor):
        """
        Using inverse bilinear interpolation based splatting
        :param frame: (h, w, c)
        :param depth: (h, w)
        :param trans_coor: (h, w, 2)
        :return: warped_frame: (h, w, c)
                 warped_depth: (j,w)
                 mask2: (h, w): True if known and False if unknown
        """
        h, w, c = frame.shape

        trans_pos = trans_coor
        trans_pos_offset = trans_pos + 1
        trans_pos_floor = np.floor(trans_pos_offset).astype('int')
        trans_pos_ceil = np.ceil(trans_pos_offset).astype('int')
        trans_pos_offset[:, :, 0] = np.clip(trans_pos_offset[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_offset[:, :, 1] = np.clip(trans_pos_offset[:, :, 1], a_min=0, a_max=h + 1)
        trans_pos_floor[:, :, 0]  = np.clip(trans_pos_floor[:, :, 0],  a_min=0, a_max=w + 1)
        trans_pos_floor[:, :, 1]  = np.clip(trans_pos_floor[:, :, 1],  a_min=0, a_max=h + 1)
        trans_pos_ceil[:, :, 0]   = np.clip(trans_pos_ceil[:, :, 0],   a_min=0, a_max=w + 1)
        trans_pos_ceil[:, :, 1]   = np.clip(trans_pos_ceil[:, :, 1],   a_min=0, a_max=h + 1)

        prox_weight_nw = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                         (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                         (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
        prox_weight_ne = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                         (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))
        prox_weight_se = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                         (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))

        sat_depth = np.clip(depth, a_min=0, a_max=100)
        log_depth = np.log(1 + sat_depth)
        depth_weights = np.exp(log_depth / log_depth.max() * 50)

        weight_nw = prox_weight_nw / depth_weights
        weight_sw = prox_weight_sw / depth_weights
        weight_ne = prox_weight_ne / depth_weights
        weight_se = prox_weight_se / depth_weights

        weight_nw_3d = weight_nw[:, :, None]
        weight_sw_3d = weight_sw[:, :, None]
        weight_ne_3d = weight_ne[:, :, None]
        weight_se_3d = weight_se[:, :, None]

        warped_image = np.zeros(shape=(h + 2, w + 2, c), dtype=np.float64)
        # warped_depth = np.zeros(shape=(h + 2, w + 2), dtype=np.float64)
        warped_weights = np.zeros(shape=(h + 2, w + 2), dtype=np.float64)

        np.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), frame * weight_nw_3d)
        np.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), frame * weight_sw_3d)
        np.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), frame * weight_ne_3d)
        np.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), frame * weight_se_3d)

        # np.add.at(warped_depth, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), depth * weight_nw)
        # np.add.at(warped_depth, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), depth * weight_sw)
        # np.add.at(warped_depth, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), depth * weight_ne)
        # np.add.at(warped_depth, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), depth * weight_se)
        np.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), weight_nw)
        np.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), weight_sw)
        np.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), weight_ne)
        np.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), weight_se)

        cropped_warped_image = warped_image[1:-1, 1:-1]
        # cropped_warped_depth = warped_depth[1:-1, 1:-1]
        cropped_weights = warped_weights[1:-1, 1:-1]

        mask = cropped_weights > 0
        with np.errstate(invalid='ignore'):
            warped_frame = np.where(mask[:, :, None], cropped_warped_image / cropped_weights[:, :, None], 0)
            warped_frame = np.round(warped_frame).astype('uint8')
            # warped_depth = np.where(mask, cropped_warped_depth / cropped_weights, 0)
            
        #return warped_frame, warped_depth, mask
        return warped_frame, mask