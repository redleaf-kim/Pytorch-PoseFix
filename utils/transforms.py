import os
import os.path as osp
import numpy as np
import cv2
import random
import time
import math
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0,):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


# 실제 어파인 변환 수행하는 부분
# t는 위의 get_affine_transform 함수를 통해 trans 매트릭스 구한다.
def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


# TODO: 뭐하는 함수지?
def get_dir(src_point, rot_rad):
    """ 
        Transformation Matrix 

        x = x * cosΘ - y * sinΘ
        y = x * cosΘ + y * sinΘ

        [ cosΘ   sinΘ   0]
        [ -sinΘ  cosΘ   0]
        [  0       0    0]
    """

    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


"""
    PoseFix의 핵심
    synthesize_pose를 만들어 모델의 입력으로 주고
    모델은 해당 pose의 refined_pose를 뱉어냄으로서 잘못된 포즈를 고쳐주는 모델을 만들수 있다.
"""


def synthesize_bottom_pose(joints, estimated_joints, near_joints, area, num_overlap, cfg, kps_symmetry):
    synth_joints = joints.copy()
    for j in range(cfg.MODEL.NUM_JOINTS):
        # in case of not annotated joints, use other models`s result and add noise
        if joints[j, 2] == 0:
            synth_joints[j] = estimated_joints[j]
            
    if random.random() > 0.3:
        l_bottoms = synth_joints[11:17:2].copy()
        r_bottoms = synth_joints[12:17:2].copy()
        
        new_bottoms = []
        for bottom in [l_bottoms, r_bottoms]:
            hip, knee, ankle = bottom[0], bottom[1], bottom[2]
            
            new_knee = knee.copy()
            new_ankle = ankle.copy()
                
            if hip[2] and knee[2]:
                frac = random.uniform(0.5, 0.8)
                diff_x = abs(knee[0]-hip[0]) * frac
                diff_y = abs(knee[1]-hip[1]) * frac
                
                if hip[0] > knee[0]: new_knee[0] = knee[0] + diff_x
                else: new_knee[0] = knee[0] - diff_x
                
                if hip[1] < knee[1]: new_knee[1] = knee[1] - diff_y
                else: new_knee[1] = knee[1] + diff_y
            
            if knee[2] and ankle[2]:
                frac = random.uniform(0.5, 0.8)
                diff_x = abs(knee[0]-ankle[0]) * frac
                diff_y = abs(knee[1]-ankle[1]) * frac
                
                if ankle[0] < knee[0]: new_ankle[0] = ankle[0] + diff_x
                else: new_ankle[0] = ankle[0] - diff_x
                
                if ankle[1] < knee[1]: new_ankle[1] = ankle[1] + diff_y
                else: new_ankle[1] = ankle[1] - diff_y
            
            new_bottoms.append(np.array([hip, new_knee, new_ankle]))
        synth_joints[11:17:2] = new_bottoms[0]
        synth_joints[12:17:2] = new_bottoms[1]
    
    
    'kps sigmas -> '
    kps_sigmas = np.array([
        .26,        # nose
        .25, .25,   # eyes
        .35, .35,   # ears
        .79, .79,   # shoulders
        .72, .72,   # elbows
        .62, .62,   # wrists
        1.07, 1.07,  # hips
        .87, .87,   # knees
        .89, .89    # ankles
    ]) / 10.0

    # TODO: 이게 뭐지?
    def get_dist_wrt_ks(ks, area):
        vars = (kps_sigmas * 2) ** 2
        return np.sqrt(-2 * area * vars * np.log(ks))

    ks_10_dist = get_dist_wrt_ks(0.10, area)
    ks_50_dist = get_dist_wrt_ks(0.50, area)
    ks_85_dist = get_dist_wrt_ks(0.85, area)

    num_valid_joint = np.sum(joints[:, 2] > 0)
        
    N = 500
    for j in [range(cfg.MODEL.NUM_JOINTS)][:11]:

        # PoseTrack does not have l_ear and r_ear. This will make input pose heatmap zero-gaussian heatmap.
        if j in cfg.ignore_kps:
            synth_joints[j] = 0
            continue

        # source keypoint position candidates to generate error on that (gt, swap, inv, swap+inv)
        coord_list = []

        # on top of gt
        gt_coord = np.expand_dims(synth_joints[j, :2], 0)
        coord_list.append(gt_coord)

        # on top of swap gt
        swap_coord = near_joints[near_joints[:, j, 2] > 0, j, :2]
        coord_list.append(swap_coord)

        # on top of inv gt, swap inv gt
        pair_exist = False
        for (q, w) in kps_symmetry:
            if j == q or j == w:
                if j == q:
                    pair_idx = w
                else:
                    pair_idx = q
                pair_exist = True
        if pair_exist and (joints[pair_idx, 2] > 0):
            inv_coord = np.expand_dims(synth_joints[pair_idx, :2], 0)
            coord_list.append(inv_coord)
        else:
            coord_list.append(np.empty([0, 2]))

        if pair_exist:
            swap_inv_coord = near_joints[near_joints[:,
                                                     pair_idx, 2] > 0, pair_idx, :2]
            coord_list.append(swap_inv_coord)
        else:
            coord_list.append(np.empty([0, 2]))

        tot_coord_list = np.concatenate(coord_list)

        assert len(coord_list) == 4

        # jitter error
        synth_jitter = np.zeros(3)
        if num_valid_joint <= 10:
            if j == 0 or (j >= 13 and j <= 16):  # nose, ankle, knee
                jitter_prob = 0.15
            elif (j >= 1 and j <= 10):  # ear, eye, upper body
                jitter_prob = 0.20
            else:  # hip
                jitter_prob = 0.25
        else:
            if j == 0 or (j >= 13 and j <= 16):  # nose, ankle, knee
                jitter_prob = 0.10
            elif (j >= 1 and j <= 10):  # ear, eye, upper body
                jitter_prob = 0.15
            else:  # hip
                jitter_prob = 0.20
        angle = np.random.uniform(0, 2*math.pi, [N])
        r = np.random.uniform(ks_85_dist[j], ks_50_dist[j], [N])
        jitter_idx = 0  # gt
        x = tot_coord_list[jitter_idx][0] + r * np.cos(angle)
        y = tot_coord_list[jitter_idx][1] + r * np.sin(angle)
        dist_mask = True
        for i in range(len(tot_coord_list)):
            if i == jitter_idx:
                continue
            dist_mask = np.logical_and(dist_mask, np.sqrt(
                (tot_coord_list[i][0] - x)**2 + (tot_coord_list[i][1] - y)**2) > r)
        x = x[dist_mask].reshape(-1)
        y = y[dist_mask].reshape(-1)
        if len(x) > 0:
            rand_idx = random.randrange(0, len(x))
            synth_jitter[0] = x[rand_idx]
            synth_jitter[1] = y[rand_idx]
            synth_jitter[2] = 1

        # miss error
        synth_miss = np.zeros(3)
        if num_valid_joint <= 5:
            if j >= 0 and j <= 4:  # face
                miss_prob = 0.15
            elif j == 5 or j == 6 or j == 15 or j == 16:  # shoulder, ankle
                miss_prob = 0.20
            else:  # other parts
                miss_prob = 0.25
        elif num_valid_joint <= 10:
            if j >= 0 and j <= 4:  # face
                miss_prob = 0.10
            elif j == 5 or j == 6 or j == 15 or j == 16:  # shoulder, ankle
                miss_prob = 0.13
            else:  # other parts
                miss_prob = 0.15
        else:
            if j >= 0 and j <= 4:  # face
                miss_prob = 0.02
            elif j == 5 or j == 6 or j == 15 or j == 16:  # shoulder, ankle
                miss_prob = 0.05
            else:  # other parts
                miss_prob = 0.10

        miss_pt_list = []
        for miss_idx in range(len(tot_coord_list)):
            angle = np.random.uniform(0, 2*math.pi, [4*N])
            r = np.random.uniform(ks_50_dist[j], ks_10_dist[j], [4*N])
            x = tot_coord_list[miss_idx][0] + r * np.cos(angle)
            y = tot_coord_list[miss_idx][1] + r * np.sin(angle)
            dist_mask = True
            for i in range(len(tot_coord_list)):
                if i == miss_idx:
                    continue
                dist_mask = np.logical_and(dist_mask, np.sqrt(
                    (tot_coord_list[i][0] - x)**2 + (tot_coord_list[i][1] - y)**2) > ks_50_dist[j])
            x = x[dist_mask].reshape(-1)
            y = y[dist_mask].reshape(-1)
            if len(x) > 0:
                if miss_idx == 0:
                    coord = np.transpose(np.vstack([x, y]), [1, 0])
                    miss_pt_list.append(coord)
                else:
                    rand_idx = np.random.choice(range(len(x)), size=len(x)//4)
                    x = np.take(x, rand_idx)
                    y = np.take(y, rand_idx)
                    coord = np.transpose(np.vstack([x, y]), [1, 0])
                    miss_pt_list.append(coord)
        if len(miss_pt_list) > 0:
            miss_pt_list = np.concatenate(miss_pt_list, axis=0).reshape(-1, 2)
            rand_idx = random.randrange(0, len(miss_pt_list))
            synth_miss[0] = miss_pt_list[rand_idx][0]
            synth_miss[1] = miss_pt_list[rand_idx][1]
            synth_miss[2] = 1

        # inversion prob
        synth_inv = np.zeros(3)
        if j <= 4:  # face
            inv_prob = 0.01
        elif j >= 5 and j <= 10:  # upper body
            inv_prob = 0.03
        else:  # lower body
            inv_prob = 0.06
        if pair_exist and joints[pair_idx, 2] > 0:
            angle = np.random.uniform(0, 2*math.pi, [N])
            r = np.random.uniform(0, ks_50_dist[j], [N])
            inv_idx = (len(coord_list[0]) + len(coord_list[1]))
            x = tot_coord_list[inv_idx][0] + r * np.cos(angle)
            y = tot_coord_list[inv_idx][1] + r * np.sin(angle)
            dist_mask = True
            for i in range(len(tot_coord_list)):
                if i == inv_idx:
                    continue
                dist_mask = np.logical_and(dist_mask, np.sqrt(
                    (tot_coord_list[i][0] - x)**2 + (tot_coord_list[i][1] - y)**2) > r)
            x = x[dist_mask].reshape(-1)
            y = y[dist_mask].reshape(-1)
            if len(x) > 0:
                rand_idx = random.randrange(0, len(x))
                synth_inv[0] = x[rand_idx]
                synth_inv[1] = y[rand_idx]
                synth_inv[2] = 1

        # swap prob
        synth_swap = np.zeros(3)
        swap_exist = (len(coord_list[1]) > 0) or (len(coord_list[3]) > 0)
        if (num_valid_joint <= 10 and num_overlap > 0) or (num_valid_joint <= 15 and num_overlap >= 3):
            if j >= 0 and j <= 4:  # face
                swap_prob = 0.02
            elif j >= 5 and j <= 10:  # upper body
                swap_prob = 0.15
            else:  # lower body
                swap_prob = 0.10
        else:
            if j >= 0 and j <= 4:  # face
                swap_prob = 0.01
            elif j >= 5 and j <= 10:  # upper body
                swap_prob = 0.06
            else:  # lower body
                swap_prob = 0.03
        if swap_exist:

            swap_pt_list = []
            for swap_idx in range(len(tot_coord_list)):
                if swap_idx == 0 or swap_idx == len(coord_list[0]) + len(coord_list[1]):
                    continue
                angle = np.random.uniform(0, 2*math.pi, [N])
                r = np.random.uniform(0, ks_50_dist[j], [N])
                x = tot_coord_list[swap_idx][0] + r * np.cos(angle)
                y = tot_coord_list[swap_idx][1] + r * np.sin(angle)
                dist_mask = True
                for i in range(len(tot_coord_list)):
                    if i == 0 or i == len(coord_list[0]) + len(coord_list[1]):
                        dist_mask = np.logical_and(dist_mask, np.sqrt(
                            (tot_coord_list[i][0] - x)**2 + (tot_coord_list[i][1] - y)**2) > r)
                x = x[dist_mask].reshape(-1)
                y = y[dist_mask].reshape(-1)
                if len(x) > 0:
                    coord = np.transpose(np.vstack([x, y]), [1, 0])
                    swap_pt_list.append(coord)
            if len(swap_pt_list) > 0:
                swap_pt_list = np.concatenate(
                    swap_pt_list, axis=0).reshape(-1, 2)
                rand_idx = random.randrange(0, len(swap_pt_list))
                synth_swap[0] = swap_pt_list[rand_idx][0]
                synth_swap[1] = swap_pt_list[rand_idx][1]
                synth_swap[2] = 1

        # good prob
        synth_good = np.zeros(3)
        good_prob = 1 - (jitter_prob + miss_prob + inv_prob + swap_prob)
        assert good_prob >= 0
        angle = np.random.uniform(0, 2*math.pi, [N//4])
        r = np.random.uniform(0, ks_85_dist[j], [N//4])
        good_idx = 0  # gt
        x = tot_coord_list[good_idx][0] + r * np.cos(angle)
        y = tot_coord_list[good_idx][1] + r * np.sin(angle)
        dist_mask = True
        for i in range(len(tot_coord_list)):
            if i == good_idx:
                continue
            dist_mask = np.logical_and(dist_mask, np.sqrt(
                (tot_coord_list[i][0] - x)**2 + (tot_coord_list[i][1] - y)**2) > r)
        x = x[dist_mask].reshape(-1)
        y = y[dist_mask].reshape(-1)
        if len(x) > 0:
            rand_idx = random.randrange(0, len(x))
            synth_good[0] = x[rand_idx]
            synth_good[1] = y[rand_idx]
            synth_good[2] = 1

        if synth_jitter[2] == 0:
            jitter_prob = 0
        if synth_inv[2] == 0:
            inv_prob = 0
        if synth_swap[2] == 0:
            swap_prob = 0
        if synth_miss[2] == 0:
            miss_prob = 0
        if synth_good[2] == 0:
            good_prob = 0

        normalizer = jitter_prob + miss_prob + inv_prob + swap_prob + good_prob
        if normalizer == 0:
            synth_joints[j] = 0
            continue

        jitter_prob = jitter_prob / normalizer
        miss_prob = miss_prob / normalizer
        inv_prob = inv_prob / normalizer
        swap_prob = swap_prob / normalizer
        good_prob = good_prob / normalizer

        prob_list = [jitter_prob, miss_prob, inv_prob, swap_prob, good_prob]
        synth_list = [synth_jitter, synth_miss,
                      synth_inv, synth_swap, synth_good]
        sampled_idx = np.random.choice(5, 1, p=prob_list)[0]
        synth_joints[j] = synth_list[sampled_idx]

        assert synth_joints[j, 2] != 0
        
    return synth_joints




def synthesize_pose(joints, estimated_joints, near_joints, area, num_overlap, cfg, kps_symmetry):
    """
        area:    bbox size
    """

    'kps sigmas -> '
    kps_sigmas = np.array([
        .26,        # nose
        .25, .25,   # eyes
        .35, .35,   # ears
        .79, .79,   # shoulders
        .72, .72,   # elbows
        .62, .62,   # wrists
        1.07, 1.07,  # hips
        .87, .87,   # knees
        .89, .89    # ankles
    ]) / 10.0

    # TODO: 이게 뭐지?
    def get_dist_wrt_ks(ks, area):
        vars = (kps_sigmas * 2) ** 2
        return np.sqrt(-2 * area * vars * np.log(ks))

    ks_10_dist = get_dist_wrt_ks(0.10, area)
    ks_50_dist = get_dist_wrt_ks(0.50, area)
    ks_85_dist = get_dist_wrt_ks(0.85, area)

    synth_joints = joints.copy()
    for j in range(cfg.MODEL.NUM_JOINTS):
        # in case of not annotated joints, use other models`s result and add noise
        if joints[j, 2] == 0:
            synth_joints[j] = estimated_joints[j]
    num_valid_joint = np.sum(joints[:, 2] > 0)
    
    # # short bottom postions
    # if random.random() > 0.3:
    #     l_bottoms = synth_joints[11:17:2].copy()
    #     r_bottoms = synth_joints[12:17:2].copy()
        
    #     new_bottoms = []
    #     for bottom in [l_bottoms, r_bottoms]:
    #         hip, knee, ankle = bottom[0], bottom[1], bottom[2]
            
    #         new_knee = knee.copy()
    #         new_ankle = ankle.copy()
                
    #         if hip[2] and knee[2]:
    #             frac = random.uniform(0.5, 0.8)
    #             diff_x = abs(knee[0]-hip[0]) * frac
    #             diff_y = abs(knee[1]-hip[1]) * frac
                
    #             if hip[0] > knee[0]: new_knee[0] = knee[0] + diff_x
    #             else: new_knee[0] = knee[0] - diff_x
                
    #             if hip[1] < knee[1]: new_knee[1] = knee[1] - diff_y
    #             else: new_knee[1] = knee[1] + diff_y
            
    #         if knee[2] and ankle[2]:
    #             frac = random.uniform(0.5, 0.8)
    #             diff_x = abs(knee[0]-ankle[0]) * frac
    #             diff_y = abs(knee[1]-ankle[1]) * frac
                
    #             if ankle[0] < knee[0]: new_ankle[0] = ankle[0] + diff_x
    #             else: new_ankle[0] = ankle[0] - diff_x
                
    #             if ankle[1] < knee[1]: new_ankle[1] = ankle[1] + diff_y
    #             else: new_ankle[1] = ankle[1] - diff_y
            
    #         new_bottoms.append(np.array([hip, new_knee, new_ankle]))
    #     synth_joints[11:17:2] = new_bottoms[0]
    #     synth_joints[12:17:2] = new_bottoms[1]


    N = 500
    for j in range(cfg.MODEL.NUM_JOINTS):

        # PoseTrack does not have l_ear and r_ear. This will make input pose heatmap zero-gaussian heatmap.
        if j in cfg.ignore_kps:
            synth_joints[j] = 0
            continue

        # source keypoint position candidates to generate error on that (gt, swap, inv, swap+inv)
        coord_list = []

        # on top of gt
        gt_coord = np.expand_dims(synth_joints[j, :2], 0)
        coord_list.append(gt_coord)

        # on top of swap gt
        swap_coord = near_joints[near_joints[:, j, 2] > 0, j, :2]
        coord_list.append(swap_coord)

        # on top of inv gt, swap inv gt
        pair_exist = False
        for (q, w) in kps_symmetry:
            if j == q or j == w:
                if j == q:
                    pair_idx = w
                else:
                    pair_idx = q
                pair_exist = True
        if pair_exist and (joints[pair_idx, 2] > 0):
            inv_coord = np.expand_dims(synth_joints[pair_idx, :2], 0)
            coord_list.append(inv_coord)
        else:
            coord_list.append(np.empty([0, 2]))

        if pair_exist:
            swap_inv_coord = near_joints[near_joints[:,
                                                     pair_idx, 2] > 0, pair_idx, :2]
            coord_list.append(swap_inv_coord)
        else:
            coord_list.append(np.empty([0, 2]))

        tot_coord_list = np.concatenate(coord_list)

        assert len(coord_list) == 4

        # jitter error
        synth_jitter = np.zeros(3)
        if num_valid_joint <= 10:
            if j == 0 or (j >= 13 and j <= 16):  # nose, ankle, knee
                jitter_prob = 0.15
            elif (j >= 1 and j <= 10):  # ear, eye, upper body
                jitter_prob = 0.20
            else:  # hip
                jitter_prob = 0.25
        else:
            if j == 0 or (j >= 13 and j <= 16):  # nose, ankle, knee
                jitter_prob = 0.10
            elif (j >= 1 and j <= 10):  # ear, eye, upper body
                jitter_prob = 0.15
            else:  # hip
                jitter_prob = 0.20
        angle = np.random.uniform(0, 2*math.pi, [N])
        r = np.random.uniform(ks_85_dist[j], ks_50_dist[j], [N])
        jitter_idx = 0  # gt
        x = tot_coord_list[jitter_idx][0] + r * np.cos(angle)
        y = tot_coord_list[jitter_idx][1] + r * np.sin(angle)
        dist_mask = True
        for i in range(len(tot_coord_list)):
            if i == jitter_idx:
                continue
            dist_mask = np.logical_and(dist_mask, np.sqrt(
                (tot_coord_list[i][0] - x)**2 + (tot_coord_list[i][1] - y)**2) > r)
        x = x[dist_mask].reshape(-1)
        y = y[dist_mask].reshape(-1)
        if len(x) > 0:
            rand_idx = random.randrange(0, len(x))
            synth_jitter[0] = x[rand_idx]
            synth_jitter[1] = y[rand_idx]
            synth_jitter[2] = 1

        # miss error
        synth_miss = np.zeros(3)
        if num_valid_joint <= 5:
            if j >= 0 and j <= 4:  # face
                miss_prob = 0.15
            elif j == 5 or j == 6 or j == 15 or j == 16:  # shoulder, ankle
                miss_prob = 0.20
            else:  # other parts
                miss_prob = 0.25
        elif num_valid_joint <= 10:
            if j >= 0 and j <= 4:  # face
                miss_prob = 0.10
            elif j == 5 or j == 6 or j == 15 or j == 16:  # shoulder, ankle
                miss_prob = 0.13
            else:  # other parts
                miss_prob = 0.15
        else:
            if j >= 0 and j <= 4:  # face
                miss_prob = 0.02
            elif j == 5 or j == 6 or j == 15 or j == 16:  # shoulder, ankle
                miss_prob = 0.05
            else:  # other parts
                miss_prob = 0.10

        miss_pt_list = []
        for miss_idx in range(len(tot_coord_list)):
            angle = np.random.uniform(0, 2*math.pi, [4*N])
            r = np.random.uniform(ks_50_dist[j], ks_10_dist[j], [4*N])
            x = tot_coord_list[miss_idx][0] + r * np.cos(angle)
            y = tot_coord_list[miss_idx][1] + r * np.sin(angle)
            dist_mask = True
            for i in range(len(tot_coord_list)):
                if i == miss_idx:
                    continue
                dist_mask = np.logical_and(dist_mask, np.sqrt(
                    (tot_coord_list[i][0] - x)**2 + (tot_coord_list[i][1] - y)**2) > ks_50_dist[j])
            x = x[dist_mask].reshape(-1)
            y = y[dist_mask].reshape(-1)
            if len(x) > 0:
                if miss_idx == 0:
                    coord = np.transpose(np.vstack([x, y]), [1, 0])
                    miss_pt_list.append(coord)
                else:
                    rand_idx = np.random.choice(range(len(x)), size=len(x)//4)
                    x = np.take(x, rand_idx)
                    y = np.take(y, rand_idx)
                    coord = np.transpose(np.vstack([x, y]), [1, 0])
                    miss_pt_list.append(coord)
        if len(miss_pt_list) > 0:
            miss_pt_list = np.concatenate(miss_pt_list, axis=0).reshape(-1, 2)
            rand_idx = random.randrange(0, len(miss_pt_list))
            synth_miss[0] = miss_pt_list[rand_idx][0]
            synth_miss[1] = miss_pt_list[rand_idx][1]
            synth_miss[2] = 1

        # inversion prob
        synth_inv = np.zeros(3)
        if j <= 4:  # face
            inv_prob = 0.01
        elif j >= 5 and j <= 10:  # upper body
            inv_prob = 0.03
        else:  # lower body
            inv_prob = 0.06
        if pair_exist and joints[pair_idx, 2] > 0:
            angle = np.random.uniform(0, 2*math.pi, [N])
            r = np.random.uniform(0, ks_50_dist[j], [N])
            inv_idx = (len(coord_list[0]) + len(coord_list[1]))
            x = tot_coord_list[inv_idx][0] + r * np.cos(angle)
            y = tot_coord_list[inv_idx][1] + r * np.sin(angle)
            dist_mask = True
            for i in range(len(tot_coord_list)):
                if i == inv_idx:
                    continue
                dist_mask = np.logical_and(dist_mask, np.sqrt(
                    (tot_coord_list[i][0] - x)**2 + (tot_coord_list[i][1] - y)**2) > r)
            x = x[dist_mask].reshape(-1)
            y = y[dist_mask].reshape(-1)
            if len(x) > 0:
                rand_idx = random.randrange(0, len(x))
                synth_inv[0] = x[rand_idx]
                synth_inv[1] = y[rand_idx]
                synth_inv[2] = 1

        # swap prob
        synth_swap = np.zeros(3)
        swap_exist = (len(coord_list[1]) > 0) or (len(coord_list[3]) > 0)
        if (num_valid_joint <= 10 and num_overlap > 0) or (num_valid_joint <= 15 and num_overlap >= 3):
            if j >= 0 and j <= 4:  # face
                swap_prob = 0.02
            elif j >= 5 and j <= 10:  # upper body
                swap_prob = 0.15
            else:  # lower body
                swap_prob = 0.10
        else:
            if j >= 0 and j <= 4:  # face
                swap_prob = 0.01
            elif j >= 5 and j <= 10:  # upper body
                swap_prob = 0.06
            else:  # lower body
                swap_prob = 0.03
        if swap_exist:

            swap_pt_list = []
            for swap_idx in range(len(tot_coord_list)):
                if swap_idx == 0 or swap_idx == len(coord_list[0]) + len(coord_list[1]):
                    continue
                angle = np.random.uniform(0, 2*math.pi, [N])
                r = np.random.uniform(0, ks_50_dist[j], [N])
                x = tot_coord_list[swap_idx][0] + r * np.cos(angle)
                y = tot_coord_list[swap_idx][1] + r * np.sin(angle)
                dist_mask = True
                for i in range(len(tot_coord_list)):
                    if i == 0 or i == len(coord_list[0]) + len(coord_list[1]):
                        dist_mask = np.logical_and(dist_mask, np.sqrt(
                            (tot_coord_list[i][0] - x)**2 + (tot_coord_list[i][1] - y)**2) > r)
                x = x[dist_mask].reshape(-1)
                y = y[dist_mask].reshape(-1)
                if len(x) > 0:
                    coord = np.transpose(np.vstack([x, y]), [1, 0])
                    swap_pt_list.append(coord)
            if len(swap_pt_list) > 0:
                swap_pt_list = np.concatenate(
                    swap_pt_list, axis=0).reshape(-1, 2)
                rand_idx = random.randrange(0, len(swap_pt_list))
                synth_swap[0] = swap_pt_list[rand_idx][0]
                synth_swap[1] = swap_pt_list[rand_idx][1]
                synth_swap[2] = 1

        # good prob
        synth_good = np.zeros(3)
        good_prob = 1 - (jitter_prob + miss_prob + inv_prob + swap_prob)
        assert good_prob >= 0
        angle = np.random.uniform(0, 2*math.pi, [N//4])
        r = np.random.uniform(0, ks_85_dist[j], [N//4])
        good_idx = 0  # gt
        x = tot_coord_list[good_idx][0] + r * np.cos(angle)
        y = tot_coord_list[good_idx][1] + r * np.sin(angle)
        dist_mask = True
        for i in range(len(tot_coord_list)):
            if i == good_idx:
                continue
            dist_mask = np.logical_and(dist_mask, np.sqrt(
                (tot_coord_list[i][0] - x)**2 + (tot_coord_list[i][1] - y)**2) > r)
        x = x[dist_mask].reshape(-1)
        y = y[dist_mask].reshape(-1)
        if len(x) > 0:
            rand_idx = random.randrange(0, len(x))
            synth_good[0] = x[rand_idx]
            synth_good[1] = y[rand_idx]
            synth_good[2] = 1

        if synth_jitter[2] == 0:
            jitter_prob = 0
        if synth_inv[2] == 0:
            inv_prob = 0
        if synth_swap[2] == 0:
            swap_prob = 0
        if synth_miss[2] == 0:
            miss_prob = 0
        if synth_good[2] == 0:
            good_prob = 0

        normalizer = jitter_prob + miss_prob + inv_prob + swap_prob + good_prob
        if normalizer == 0:
            synth_joints[j] = 0
            continue

        jitter_prob = jitter_prob / normalizer
        miss_prob = miss_prob / normalizer
        inv_prob = inv_prob / normalizer
        swap_prob = swap_prob / normalizer
        good_prob = good_prob / normalizer

        prob_list = [jitter_prob, miss_prob, inv_prob, swap_prob, good_prob]
        synth_list = [synth_jitter, synth_miss,
                      synth_inv, synth_swap, synth_good]
        sampled_idx = np.random.choice(5, 1, p=prob_list)[0]
        synth_joints[j] = synth_list[sampled_idx]

        assert synth_joints[j, 2] != 0

    return synth_joints
