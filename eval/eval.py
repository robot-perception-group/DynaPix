#!/usr/bin/python
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from associate import associate, read_file_list
from evaluate_ate import align
import evaluate_rpe as rpe

def evaluate_ate(gtruth_file, estimate_file, offset, max_difference, start_time, end_time):
    """
    Computes the absolute trajectory error from the ground truth trajectory
    and the estimated trajectory.
    
    Input:
    gtruth_file -- File name of the ground truth trajectory. 
    estimate_file -- File name of the estimated trajectory. 
    offset -- time offset between ground truth and estimated trajectory
    max_difference -- search radius for matching entries
    start_time -- start time for evaluation
    end_time -- end time for evaluation
    
    Output:
    (ate, ate_list) -- tuple of absolute trajectory error (ATE) and a list of individual errors
    """
    # read ground truth and estimated trajectory
    gtruth = read_file_list(gtruth_file)
    estimate = read_file_list(estimate_file)
    
    # compute matches
    matches = associate(gtruth, estimate, float(offset), float(max_difference), start_time, end_time)
    if len(matches)<2:
        sys.exit("Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! Did you choose the correct sequence?")
    
    gtruth_xyz    = np.matrix([[float(value) for value in gtruth[a][0:3]] for a,b in matches]).transpose()
    estimate_xyz = np.matrix([[float(value) for value in estimate[b][0:3]] for a,b in matches]).transpose()
    
    rot, trans, trans_error = align(estimate_xyz, gtruth_xyz)
    
    estimate_xyz_aligned = rot * estimate_xyz + trans
    
    gtruth_stamps = list(gtruth)
    gtruth_stamps.sort()
    gtruth_xyz_full = np.matrix([[float(value) for value in gtruth[b][0:3]] for b in gtruth_stamps]).transpose()
    
    estimate_stamps = list(estimate)
    estimate_stamps.sort()
    estimate_xyz_full = np.matrix([[float(value) for value in estimate[b][0:3]] for b in estimate_stamps]).transpose()
    estimate_xyz_full_aligned = rot * estimate_xyz_full + trans

    return trans_error

def evaluate_rpe(gtruth_file, estimate_file, max_pairs=10000, offset=0.00, scale=1.0, fixed_delta=False, delta=1.00, delta_unit='s'):
    """
    Computes the relative pose error from the ground truth trajectory
    and the estimated trajectory.

    Args:
        gtruth_file (_type_): _description_
        estimate_file (_type_): _description_
        max_pairs (int, optional): _description_. Defaults to 10000.
        offset (int, optional): _description_. Defaults to 0.
        delta (float, optional): _description_. Defaults to 1.0.
        delta_unit (str, optional): _description_. Defaults to 's'.

    Returns:
        _type_: _description_
    """
    # read ground truth and estimated trajectory
    traj_gt = rpe.read_trajectory(gtruth_file)
    traj_est = rpe.read_trajectory(estimate_file)
    
    result = rpe.evaluate_trajectory(traj_gt,
                                traj_est,
                                int(max_pairs),
                                fixed_delta,
                                float(delta),
                                delta_unit,
                                float(offset),
                                float(scale))

    stamps = np.array(result)[:, 0]
    trans_error = np.array(result)[:, 4]
    rot_error = np.array(result)[:, 5]

    trans_error_rmse = np.sqrt(np.dot(trans_error, trans_error) / len(trans_error))
    rot_error_rmse = np.sqrt(np.dot(rot_error, rot_error) / len(rot_error)) * 180.0 / np.pi
    
    return trans_error_rmse, rot_error_rmse

def evaluate_time(estimate_file, reference_file, start_time, end_time):
    """
    Computes the missing time from estimated trajectory.
    
    Input:
    estimate_file -- File name of the estimated trajectory
    start_time -- Start time for evaluation
    end_time -- End time for evaluation
    
    Output:
    t_miss -- Missing Time

    """
    # Load Reference Timestamps
    with open(reference_file) as f:
        data = f.readlines()
        ts_gt = []
        for d in data:
            t = float((d.split(' ')[0]).replace(",","."))
            ts_gt.append(t)

    # Load Estimated Timestamps
    with open(estimate_file) as f:
        # load estimated poses
        data = f.readlines()
        ts = []
        for d in data:
            t = float((d.split(' ')[0]).replace(",","."))
            ts.append(t)
    
    # time insterval (number of timestamps - 1)
    Ts_missing = np.ones(len(ts_gt)-1, dtype=bool)
    
    # Find the missing time interval
    for idx, t in enumerate(ts_gt):        
        # Start from the "start_time"
        if t < start_time or t > end_time:
            continue
        
        # TODO: release the tolerance
        if t not in ts:
            if idx == 0:
                Ts_missing[0] = 0
            elif idx == len(ts_gt)-1:
                Ts_missing[-1] = 0
            else:
                Ts_missing[idx-1] = 0
                Ts_missing[idx] = 0
    
    # Compute the missing time
    t_miss = 0 # missing time
    ts_diff = np.diff(ts_gt)
    
    for i in range(len(Ts_missing)):
        if not Ts_missing[i]:
            t_miss += ts_diff[i]

    return t_miss


if __name__ == '__main__':
    # parse command line
    parser = argparse.ArgumentParser(description='''Multi-Trajectories Evaluation''')
    parser.add_argument('gt_file', help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('ref_file', help='Trajecotry Completeness based on Ground-Truth Timestamps', default='rgbd_assoc.txt')
    parser.add_argument('exp_path', help='Folder of Experiments (contain multiple trajectories)')
    # vairable input
    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',default=0.0)
    parser.add_argument('--scale', help='scaling factor for the second trajectory (default: 1.0)',default=1.0)
    parser.add_argument('--max_difference', help='maximally allowed time difference for matching entries (default: 0.02)',default=0.02)
    parser.add_argument('--start_time', type=float, default=0.0)
    parser.add_argument('--end_time',   type=float, default=60.0)
    
    parser.add_argument('--rpe', help='compute relative pose error', action='store_true')
    parser.add_argument('--plot', help='plot the first and the aligned second trajectory to an image (format: png)')
    parser.add_argument('--save', help='save aligned second trajectory to disk (format: stamp2 x2 y2 z2)')
    parser.add_argument('--save_associations', help='save associated first and aligned second trajectory to disk (format: stamp1 x1 y1 z1 stamp2 x2 y2 z2)')
    args = parser.parse_args()
    
    # Define the stop timestamp
    with open(args.gt_file) as f:
        data = f.readlines()
        ts_gt_start = float((data[0].split(' ')[0]).replace(",","."))
        ts_gt_final = float((data[-1].split(' ')[0]).replace(",","."))
    
    ts_start = max(args.start_time, ts_gt_start)  # initilize the first timestamp
    ts_end = min(args.end_time, ts_gt_final) # initilize the final timestamp
        
    PATH = args.exp_path
    EXPS = os.listdir(PATH) # get all the trajectories
    
    keys = os.listdir(PATH)
    
    # Load Evaluation Result
    if 'ATE.npy' in EXPS:
        ATE = np.load(os.path.join(PATH, 'ATE.npy'), allow_pickle=True).item()
        MissingTime = np.load(os.path.join(PATH, 'MissingTime.npy'), allow_pickle=True).item()
        evo = np.load(os.path.join(PATH, 'evo.npy'), allow_pickle=True).item()
    else:
        evo = {}
        ATE = {}
        MissingTime = {}
    
    for key in keys:
        if key not in ATE.keys():
            # initialize evaluation
            exp = key
            evo[exp] = {}
            
            exp_fn = os.path.join(PATH, exp)
            if not os.path.isdir(exp_fn): continue
            
            print('\n========== %s  ==========' % exp)
            trajs = [traj for traj in os.listdir(exp_fn) if ('.txt' in traj)]
            trajs = trajs# get all the trajectories
            trajs.sort()
        
            n_traj = len(trajs) # number of trajectories

            # Initialize trajectory result
            ATE[exp] = []
            MissingTime[exp] = []
            for traj in trajs:
                # Initialize the Trajectory
                evo[exp][traj] = {}
                
                # Load Trajectory
                traj_fn = os.path.join(exp_fn, traj)
                
                # compute absolute trajectory error
                trans_error = evaluate_ate(args.gt_file, traj_fn, args.offset, args.max_difference, ts_start, ts_end)
                
                # compute missing time
                t_miss = evaluate_time(traj_fn, args.ref_file, ts_start, ts_end)
                
                evo[exp][traj]['ATE'] = np.sqrt(np.dot(trans_error,trans_error) / len(trans_error))
                evo[exp][traj]['MISSING_TIME'] = t_miss
                
                ATE[exp].append(evo[exp][traj]['ATE'])
                MissingTime[exp].append(evo[exp][traj]['MISSING_TIME'])
                
                # compute relative trajectory error
                if args.rpe:
                    trans_relative_error, rot_relative_error = evaluate_rpe(args.gt_file, traj_fn, offset=args.offset, scale=args.scale, fixed_delta=False, delta=1.0, delta_unit='s')
                    evo[exp][traj]['RPE_TRANS'] = trans_relative_error
                    evo[exp][traj]['RPE_ROT'] = rot_relative_error
                    print('%s  - ATE : %.6f , Missing Time : %.6f , RPE-TRANS: %.6f, RPE-ROT: %.6f\n' \
                        % (traj, evo[exp][traj]['ATE'], evo[exp][traj]['MISSING_TIME'], evo[exp][traj]['RPE_TRANS'],evo[exp][traj]['RPE_ROT']))
                    # direct output
                    # print('%.6f , %.6f , %.6f, %.6f' \
                        # % (evo[exp][traj]['ATE'], evo[exp][traj]['RPE_TRANS'],evo[exp][traj]['RPE_ROT'], evo[exp][traj]['MISSING_TIME']))    
                else:
                    print('%s  - ATE : %.6f , Missing Time : %.6f' % (traj, evo[exp][traj]['ATE'], evo[exp][traj]['MISSING_TIME']))
                    # direct output
                    # print('%.6f , %.6f' % (evo[exp][traj]['ATE'], evo[exp][traj]['MISSING_TIME'])) 
            # np.save(os.path.join(PATH, 'ATE.npy'), ATE)
            # np.save(os.path.join(PATH, 'MissingTime.npy'), MissingTime)
            # np.save(os.path.join(PATH, 'evo.npy'), evo)