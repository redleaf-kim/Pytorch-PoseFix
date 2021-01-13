# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import random
import logging
import math
import cv2
import numpy as np
from PIL import Image

import torch

import pickle
import zipfile
from collections import defaultdict
from collections import OrderedDict
import matplotlib.pyplot as plt

import json_tricks as json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


from .base import BaseDataset
from nms.nms import oks_nms
from utils.transforms import get_affine_transform, affine_transform, synthesize_pose, synthesize_bottom_pose
from config import config

logging.basicConfig(level=logging.INFO, filename=config.LOG_DIR,
                    format="%(asctime)s => %(message)s")
# logging.basicConfig(level=logging.INFO, format="%(name)s :: %(asctime)s => %(message)s")


class COCODataset(BaseDataset):
    '''
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
        "skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    '''

    def __init__(self, cfg, root, image_set, is_train, is_eval=False, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        # TODO 나중에 파라미터로 바꿀 것
        self.test_on_trainset_path = "./data/test_on_trainpath/keypoints_train2017_results.json"
        self.input_pose_path = "./data/input_pose_path/keypoints_valid2017_results.json"

        annot_name = "train" if is_train else "valid"
        self.annot_path = f"./data/annotations/person_keypoints_{annot_name}.json"

        self.cfg = cfg
        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        self.coco = COCO(self.annot_path)
        self.transform = transform

        self.num_joints = 17
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
        self.parent_ids = None

        loading_time = time.time()
        logging.info(f'loading {annot_name} datas...')

        self.eval = is_eval
        if not is_eval:
            if is_train:
                self.data = self._make_train_data()
            else:
                self.data = self._make_valid_data()
            logging.info('load {} samples taken {}s\n'.format(
                len(self.data), time.time()-loading_time))
        else:
            self.data = self.input_pose_load(self.coco)

    def input_pose_load(self, annot):
        gt_img_id = self.load_imgid(annot)

        with open(self.input_pose_path, 'r') as f:
            input_pose = json.load(f)

        # for i in range(len(input_pose)):
            # input_pose[i]['score'] = np.mean(input_pose[i]['scores'])

        input_pose = [i for i in input_pose if i['image_id'] in gt_img_id]
        input_pose = [i for i in input_pose if i['category_id'] == 1]
        input_pose = [i for i in input_pose if i['score'] > 0]
        input_pose.sort(key=lambda x: (
            x['image_id'], x['score']), reverse=True)

        img_id = []
        for i in input_pose:
            img_id.append(i['image_id'])
        imgname = self.imgid_to_imgname(annot, img_id)
        for i in range(len(input_pose)):
            input_pose[i]['imgpath'] = imgname[i]

        # bbox generate
        for i in range(len(input_pose)):
            input_pose[i]['estimated_joints'] = input_pose[i]['keypoints']
            input_pose[i]['estimated_score'] = input_pose[i]['score']
            del input_pose[i]['keypoints']
            del input_pose[i]['score']
            # del input_pose[i]['scores']

            coords = np.array(input_pose[i]['estimated_joints']).reshape(
                self.cfg.MODEL.NUM_JOINTS, 3)
            coords = np.delete(coords, self.cfg.ignore_kps, axis=0)

            xmin = np.min(coords[:, 0])
            xmax = np.max(coords[:, 0])
            width = xmax - xmin if xmax > xmin else 20
            center = (xmin + xmax)/2.
            xmin = center - width/2.*1.2
            xmax = center + width/2.*1.2

            ymin = np.min(coords[:, 1])
            ymax = np.max(coords[:, 1])
            height = ymax - ymin if ymax > ymin else 20
            center = (ymin + ymax)/2.
            ymin = center - height/2.*1.2
            ymax = center + height/2.*1.2

            input_pose[i]['bbox'] = [xmin, ymin, xmax-xmin, ymax-ymin]

        return input_pose

    def readfromZip(self, name, is_train=True):
        folder = "train" if is_train else "valid"
        file = "train2017" if is_train else "valid2017"
        zipFile = os.path.join(self.root, "images", folder, file + ".zip")

        with zipfile.ZipFile(zipFile, 'r') as zfile:
            data = zfile.read(os.path.join(file, name))

        img = cv2.imdecode(np.frombuffer(data, np.uint8),
                           cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        return img

    def randomErase(self, img, estimated_joints, transform):
        assert transform is not None
        
        if not all(estimated_joints[11:17, 2]): 
            return img
        else:
            copy_img = img.copy()
            roi_joints = estimated_joints[11:17, :2]
            
            min_x, max_x = min(roi_joints[:, 0]), max(roi_joints[:, 0]) * 1.2
            min_y, max_y = min(roi_joints[:, 1]), max(roi_joints[:, 1]) * 1.2
                
                
            # h = (max_y - min_y) * random.uniform(0.8, 1.2)
            # w = (max_x - min_x) * random.uniform(0.8, 1.2)
            
            # random_x = random.uniform(min_x*0.95, max_x*0.95)
            # random_y = random.uniform(min_y*0.95, max_y*0.95)
            
            roi_img = torch.from_numpy(copy_img[int(min_y):int(max_y), int(min_x):int(max_x), :]).float().permute(2, 0, 1)
            roi_img = transform(roi_img)
            copy_img[int(min_y):int(max_y),  int(min_x):int(max_x), :] = roi_img.permute(1, 2, 0).numpy()
                
            return copy_img
            
        

    def __getitem__(self, idx):
        data = self.data[idx]

        imgpath = data['imgpath']
        if self.data_format == "zip":
            img = self.readfromZip(imgpath)
        else:
            folder = 'train' if self.is_train else 'valid'
            path = os.path.join(self.root, 'images', folder, imgpath)
            img = cv2.imread(path, cv2.IMREAD_COLOR |
                             cv2.IMREAD_IGNORE_ORIENTATION)

        if img is None:
            raise Exception(
                f"cannot read image file {imgpath} form {self.root}")

        bbox = np.array(data['bbox']).astype(np.float32)

        x, y, w, h = bbox
        aspect_ratio = self.cfg.MODEL.IMAGE_SIZE[1] / \
            self.cfg.MODEL.IMAGE_SIZE[0]
        centre = np.array([x+w*.5, y+h*.5])

        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array([w, h]) * 1.25
        rotation = 0

        if self.is_train:
            joints = np.array(data['joints']).reshape(-1,self.cfg.MODEL.NUM_JOINTS, 3)
            estimated_joints = np.array(data['estimated_joints']).reshape(-1, self.cfg.MODEL.NUM_JOINTS, 3)
            near_joints = np.array(data['near_joints']).reshape(-1, self.cfg.MODEL.NUM_JOINTS, 3)
            total_joints = np.concatenate([joints, estimated_joints, near_joints], axis=0)

            # 데이터 augmentation
            scale = scale * np.clip(np.random.randn()*self.cfg.DATASET.SCALE_FACTOR+1,
                                    1-self.cfg.DATASET.SCALE_FACTOR, 1+self.cfg.DATASET.SCALE_FACTOR)
            rotation = np.clip(np.random.randn()*self.cfg.DATASET.ROT_FACTOR, -self.cfg.DATASET.ROT_FACTOR*2, self.cfg.DATASET.ROT_FACTOR*2) \
                if random.random() <= 0.6 else 0

            # lr flipping
            if np.random.random() <= 0.5:
                img = img[:, ::-1, :]
                centre[0] = img.shape[1] - 1 - centre[0]

                total_joints[:, :, 0] = img.shape[1] - \
                    1 - total_joints[:, :, 0]
                for (q, w) in self.flip_pairs:
                    total_joints_q, total_joints_w = total_joints[:, q, :].copy(
                    ), total_joints[:, w, :].copy()
                    total_joints[:, w, :], total_joints[:, q,
                                                        :] = total_joints_q, total_joints_w

            trans = get_affine_transform(centre, scale, rotation, (self.cfg.MODEL.IMAGE_SIZE[1], self.cfg.MODEL.IMAGE_SIZE[0]))
            cropped_img = cv2.warpAffine(img, trans, (self.cfg.MODEL.IMAGE_SIZE[1], self.cfg.MODEL.IMAGE_SIZE[0]), flags=cv2.INTER_LINEAR)
            cropped_img = self.normalize_input(cropped_img)

            for i in range(len(total_joints)):
                for j in range(self.cfg.MODEL.NUM_JOINTS):
                    if total_joints[i, j, 2] > 0:
                        total_joints[i, j, :2] = affine_transform(
                            total_joints[i, j, :2], trans)
                        total_joints[i, j, 2] *= ((total_joints[i, j, 0] >= 0) & (total_joints[i, j, 0] < self.cfg.MODEL.IMAGE_SIZE[1]) & (
                            total_joints[i, j, 1] >= 0) & (total_joints[i, j, 1] < self.cfg.MODEL.IMAGE_SIZE[0]))

            joints = total_joints[0]
            estimated_joints = total_joints[1]
            near_joints = total_joints[2:]

            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[0] + \
                bbox[2], bbox[1]+bbox[3]
            pt1 = affine_transform(np.array([xmin, ymin]), trans)
            pt2 = affine_transform(np.array([xmax, ymin]), trans)
            pt3 = affine_transform(np.array([xmax, ymax]), trans)
            area = math.sqrt(pow(pt2[0] - pt1[0], 2) + pow(pt2[1] - pt1[1], 2)) * \
                math.sqrt(pow(pt3[0] - pt2[0], 2) + pow(pt3[1] - pt2[1], 2))

            # input pose synthesize
            # if config.TRAIN.SHORT_BOTTOM and random.random() > 0.5:
            #     short_bottom_joints = synthesize_bottom_pose(joints, estimated_joints, self.cfg)
            #     estimated_joints[11:17:2] = short_bottom_joints[0]
            #     estimated_joints[12:17:2] = short_bottom_joints[1]
            
            
            # TODO: 에러가 난다... 원래 괜찮았었는데 무엇이 문제인지 잘 모르겠다....
            # if np.random.random() <= 0.3:
            #     synth_joints = synthesize_bottom_pose(
            #         joints, estimated_joints, near_joints, area, data['overlap'], self.cfg, self.flip_pairs)
            #     erase_cropped_img = self.randomErase(cropped_img, joints, self.transform)
            # else:
            synth_joints = synthesize_pose(
                joints, estimated_joints, near_joints, area, data['overlap'], self.cfg, self.flip_pairs)
            erase_cropped_img = cropped_img.copy()

            target_coord = joints[:, :2]
            target_valid = joints[:, 2]
            input_pose_coord = synth_joints[:, :2]
            input_pose_valid = synth_joints[:, 2]

            
            # for debug
            vis = False
            if vis:
                os.makedirs('debug', exist_ok=True)
                
                filename = str(random.randrange(1,500))
                tmpimg = cropped_img.astype(np.float32).copy()
                tmpimg = self.denormalize_input(tmpimg)
                tmpimg = tmpimg.astype(np.uint8).copy()
                tmpkps = np.zeros((3, self.cfg.MODEL.NUM_JOINTS))
                tmpkps[:2,:] = target_coord.transpose(1,0)
                tmpkps[2,:] = target_valid
                tmpimg = self.vis_keypoints(tmpimg, tmpkps)
                cv2.imwrite(os.path.join('debug', filename + '_gt.jpg'), tmpimg)
    
                tmpimg = cropped_img.astype(np.float32).copy()
                tmpimg = self.denormalize_input(tmpimg)
                tmpimg = tmpimg.astype(np.uint8).copy()
                tmpkps = np.zeros((3,self.cfg.MODEL.NUM_JOINTS))
                tmpkps[:2,:] = input_pose_coord.transpose(1,0)
                tmpkps[2,:] = input_pose_valid
                tmpimg = self.vis_keypoints(tmpimg, tmpkps)
                cv2.imwrite(os.path.join('debug', filename + '_input_pose.jpg'), tmpimg)


            return [torch.from_numpy(cropped_img).float().permute(2, 0, 1),
                    torch.from_numpy(erase_cropped_img).float().permute(2, 0, 1),
                    target_coord,
                    input_pose_coord,
                    (target_valid > 0),
                    (input_pose_valid > 0)]
        else:
            trans = get_affine_transform(centre, scale, rotation, (self.cfg.MODEL.IMAGE_SIZE[1], self.cfg.MODEL.IMAGE_SIZE[0]))
            cropped_img = cv2.warpAffine(img, trans, (self.cfg.MODEL.IMAGE_SIZE[1], self.cfg.MODEL.IMAGE_SIZE[0]), flags=cv2.INTER_LINEAR)
            cropped_img = self.normalize_input(cropped_img)

            estimated_joints = np.array(data['estimated_joints']).reshape(self.cfg.MODEL.NUM_JOINTS, 3)
            for j in range(self.cfg.MODEL.NUM_JOINTS):
                if estimated_joints[j, 2] > 0:
                    estimated_joints[j, :2] = affine_transform(
                        estimated_joints[j, :2], trans)
                    estimated_joints[j, 2] *= ((estimated_joints[j, 0] >= 0) & (estimated_joints[j, 0] < self.cfg.MODEL.IMAGE_SIZE[1]) & (
                        estimated_joints[j, 1] >= 0) & (estimated_joints[j, 1] < self.cfg.MODEL.IMAGE_SIZE[0]))


            input_pose_coord = estimated_joints[:, :2]
            input_pose_valid = np.array([1 if i not in self.cfg.ignore_kps else 0 for i in range(self.cfg.MODEL.NUM_JOINTS)])
            input_pose_score = estimated_joints[:, 2]

            crop_info = np.asarray([centre[0]-scale[0]*0.5, centre[1] -
                                    scale[1]*0.5, centre[0]+scale[0]*0.5, centre[1]+scale[1]*0.5])

            returns = [torch.from_numpy(cropped_img).float().permute(2, 0, 1),
                       input_pose_coord,
                       input_pose_valid,
                       input_pose_score,
                       crop_info
                       ]

            if not self.eval:
                return returns
            else:
                return returns, data

    def vis_keypoints(self, img, kps, kp_thresh=0.4, alpha=1):
        """
            Visualizes keypoints (adapted from vis_one_image).
            kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
        """

        # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
        kps_lines = [
            (1, 2), (0, 1), (0, 2), (2, 4), (1, 3),
            (6, 8), (8, 10), (5, 7), (7, 9), (12, 14),
            (14, 16), (11, 13), (13, 15), (5, 6), (11, 12)
        ]

        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

        # Perform the drawing on a copy of the image, to allow for blending.
        kp_mask = np.copy(img)

        # Draw mid shoulder / mid hip first for better visualization.
        mid_shoulder = (
            kps[:2, 5] +
            kps[:2, 6]) / 2.0
        sc_mid_shoulder = np.minimum(
            kps[2, 5],
            kps[2, 6])
        mid_hip = (
            kps[:2, 11] +
            kps[:2, 12]) / 2.0
        sc_mid_hip = np.minimum(
            kps[2, 11],
            kps[2, 12])

        nose_idx = 0
        if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
            cv2.line(
                kp_mask, tuple(mid_shoulder.astype(np.int32)), tuple(
                    kps[:2, nose_idx].astype(np.int32)),
                color=colors[len(kps_lines)], thickness=2, lineType=cv2.LINE_AA)
        if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
            cv2.line(
                kp_mask, tuple(mid_shoulder.astype(np.int32)), tuple(
                    mid_hip.astype(np.int32)),
                color=colors[len(kps_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

        # Draw the keypoints.
        for l in range(len(kps_lines)):
            i1 = kps_lines[l][0]
            i2 = kps_lines[l][1]
            p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
            p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
            if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                cv2.line(
                    kp_mask, p1, p2,
                    color=colors[l], thickness=2, lineType=cv2.LINE_AA)
            if kps[2, i1] > kp_thresh:
                cv2.circle(
                    kp_mask, p1,
                    radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            if kps[2, i2] > kp_thresh:
                cv2.circle(
                    kp_mask, p2,
                    radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

        # Blend the keypoints.
        return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

    def normalize_input(self, img):
        return img - self.cfg.pixel_means

    def denormalize_input(self, img):
        return img + self.cfg.pixel_means

    def _make_train_data(self):
        train_data = []
        for aid in self.coco.anns.keys():
            ann = self.coco.anns[aid]
            imgname = self.coco.imgs[ann['image_id']]['file_name']
            joints = ann['keypoints']

            if (ann["image_id"] not in self.coco.imgs) or ann['iscrowd'] or (np.sum(joints[2::3]) < 10) or (ann['num_keypoints'] < 10):
                continue

            x, y, w, h = ann['bbox']
            img = self.coco.loadImgs(ann['image_id'])[0]
            width, height = img['width'], img['height']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if ann['area'] > 0 and x2 >= x1 and y2 >= y1:
                bbox = [x1, y1, x2-x1, y2-y1]
            else:
                continue

            data = dict(image_id=ann['image_id'],
                        imgpath=imgname, bbox=bbox, joints=joints)
            train_data.append(data)

        with open(self.test_on_trainset_path, 'r') as f:
            test_on_trainset = json.load(f)

        train_data = sorted(train_data, key=lambda k: k['image_id'])
        test_on_trainset = sorted(
            test_on_trainset, key=lambda k: k['image_id'])

        ################################################################
        # 훈련데이터 image_id별로 클러스터링                           #
        ################################################################
        cur_img_id = train_data[0]['image_id']

        data_gt = []
        data_gt_per_img = []
        for i in range(len(train_data)):
            if train_data[i]['image_id'] == cur_img_id:
                data_gt_per_img.append(train_data[i])
            else:
                data_gt.append(data_gt_per_img)
                cur_img_id = train_data[i]['image_id']
                data_gt_per_img = [train_data[i]]
        if len(data_gt_per_img) > 0:
            data_gt.append(data_gt_per_img)

        ################################################################
        # https://github.com/microsoft/human-pose-estimation.pytorch   #
        # pose_net으로 예측한 결과 데이터 image_id별 클러스터링        #
        ################################################################
        cur_img_id = test_on_trainset[0]['image_id']
        data_out = []
        data_out_per_img = []
        for i in range(len(test_on_trainset)):
            if test_on_trainset[i]['image_id'] == cur_img_id:
                data_out_per_img.append(test_on_trainset[i])
            else:
                data_out.append(data_out_per_img)
                cur_img_id = test_on_trainset[i]['image_id']
                data_out_per_img = [test_on_trainset[i]]
        if len(data_out_per_img) > 0:
            data_out.append(data_out_per_img)

        ################################################################
        #                   FP 데이터셋 제거 해주기                    #
        ################################################################
        i = 0
        j = 0
        aligned_data_out = []
        while True:
            gt_img_id = data_gt[i][0]['image_id']
            out_img_id = data_out[j][0]['image_id']

            # image_id가 다르면 작은 값을 하나씩 증가
            # 같다면 aligned_data_out에 추가해주고 두개 값을 하나씩 증가
            if gt_img_id > out_img_id:
                j += 1
            elif gt_img_id < out_img_id:
                i += 1
            else:
                aligned_data_out.append(data_out[j])

                i += 1
                j += 1

            if j == len(data_out) or i == len(data_gt):
                break
        data_out = aligned_data_out

        ################################################################
        #                   FN 데이터셋 추가 해주기                    #
        ################################################################
        j = 0
        aligned_data_out = []
        for i in range(len(data_gt)):
            gt_img_id = data_gt[i][0]['image_id']
            out_img_id = data_out[j][0]['image_id']

            if gt_img_id == out_img_id:
                aligned_data_out.append(data_out[j])
                j = j + 1
            else:
                aligned_data_out.append([])

            if j == len(data_out):
                break
        data_out = aligned_data_out

        # they should contain annotations from all the images
        assert (len(data_gt) == len(data_out))

        # for each img
        # TODO: 뭘 하는 건지 파악해보자
        drop_list = []
        for i in range(len(data_gt)):
            bbox_out_per_img = np.zeros((len(data_out[i]), 4))
            joint_out_per_img = np.zeros(
                (len(data_out[i]), self.cfg.MODEL.NUM_JOINTS*3))

            if not len(data_gt[i]) == len(data_out[i]):
                drop_list.append(i)
                continue

            # for each data_out in an img
            aspect_ratio = self.cfg.MODEL.IMAGE_SIZE[1] / \
                self.cfg.MODEL.IMAGE_SIZE[0]
            for j in range(len(data_out[i])):
                s = data_out[i][j]['scale']
                c = data_out[i][j]['center']

                if c[0] != -1:
                    w = s[0] / 1.25 * 200
                    h = s[1] / 1.25 * 200
                else:
                    w = s[0] * 200
                    h = s[1] * 200

                x = c[0] - w*0.5
                y = c[1] - h*0.5

                # bbox = data_out[i][j]['bbox'] #x, y, width, height
                bbox = x, y, w, h
                joint = data_out[i][j]['keypoints']
                bbox_out_per_img[j, :] = bbox
                joint_out_per_img[j, :] = joint

            # for each gt in an img
            for j in range(len(data_gt[i])):
                # x, y, width, height
                bbox_gt = np.array(data_gt[i][j]['bbox'])
                joint_gt = np.array(data_gt[i][j]['joints'])

                # IoU calculate with detection outputs of other methods
                iou = self.compute_iou(bbox_gt.reshape(1, 4), bbox_out_per_img)
                out_idx = np.argmax(iou)
                data_gt[i][j]['estimated_joints'] = [
                    joint_out_per_img[out_idx, :]]

                # for swap
                num_overlap = 0
                near_joints = []
                for k in range(len(data_gt[i])):
                    bbox_gt_k = np.array(data_gt[i][k]['bbox'])
                    iou_with_gt_k = self.compute_iou(
                        bbox_gt.reshape(1, 4), bbox_gt_k.reshape(1, 4))
                    if k == j or iou_with_gt_k < 0.1:
                        continue
                    num_overlap += 1
                    near_joints.append(np.array(data_gt[i][k]['joints']).reshape(
                        self.cfg.MODEL.NUM_JOINTS, 3))
                data_gt[i][j]['overlap'] = num_overlap
                if num_overlap > 0:
                    data_gt[i][j]['near_joints'] = near_joints
                else:
                    data_gt[i][j]['near_joints'] = [
                        np.zeros([self.cfg.MODEL.NUM_JOINTS, 3])]

        # flatten data_gt
        data_gt = [data_gt[i]
                   for i in range(len(data_gt)) if i not in drop_list]
        return [y for x in data_gt for y in x]

    def load_imgid(self, annot):
        return annot.imgs

    def imgid_to_imgname(self, annot, imgid):
        imgs = annot.loadImgs(imgid)
        folder = 'train' if self.is_train else 'valid'
        imgpath = os.path.join(self.root, 'images', folder)
        imgname = [os.path.join(imgpath, i['file_name']) for i in imgs]
        return imgname

    def _make_valid_data(self):
        val_data = []
        for aid in self.coco.anns.keys():
            ann = self.coco.anns[aid]
            if ann['image_id'] not in self.coco.imgs:
                continue
            imgname = self.coco.imgs[ann['image_id']]['file_name']
            joints = ann['keypoints']
            bbox = ann['bbox']

            data = dict(
                image_id=ann['image_id'], imgpath=imgname, bbox=bbox, joints=joints, score=1)
            val_data.append(data)

        gt_img_id = self.load_imgid(self.coco)

        with open(self.input_pose_path, 'r') as f:
            input_pose = json.load(f)

        input_pose = [i for i in input_pose if i['image_id'] in gt_img_id]
        input_pose = [i for i in input_pose if i['category_id'] == 1]
        input_pose = [i for i in input_pose if i['score'] > 0]
        input_pose.sort(key=lambda x: (
            x['image_id'], x['score']), reverse=True)

        img_id = []
        for i in input_pose:
            img_id.append(i['image_id'])
        imgname = self.imgid_to_imgname(self.coco, img_id, val_data)
        for i in range(len(input_pose)):
            input_pose[i]['imgpath'] = imgname[i]

        # bbox generate
        for i in range(len(input_pose)):
            input_pose[i]['estimated_joints'] = input_pose[i]['keypoints']
            input_pose[i]['estimated_score'] = input_pose[i]['score']
            del input_pose[i]['keypoints']
            del input_pose[i]['score']

            coords = np.array(input_pose[i]['estimated_joints']).reshape(
                self.cfg.MODEL.NUM_JOINTS, 3)
            coords = np.delete(coords, self.cfg.ignore_kps, axis=0)

            xmin = np.min(coords[:, 0])
            xmax = np.max(coords[:, 0])
            width = xmax - xmin if xmax > xmin else 20
            center = (xmin + xmax)/2.
            xmin = center - width/2.*1.2
            xmax = center + width/2.*1.2

            ymin = np.min(coords[:, 1])
            ymax = np.max(coords[:, 1])
            height = ymax - ymin if ymax > ymin else 20
            center = (ymin + ymax)/2.
            ymin = center - height/2.*1.2
            ymax = center + height/2.*1.2

            input_pose[i]['bbox'] = [xmin, ymin, xmax-xmin, ymax-ymin]

        return input_pose

    def compute_iou(self, src_roi, dst_roi):
        # IoU calculate with GTs
        xmin = np.maximum(dst_roi[:, 0], src_roi[:, 0])
        ymin = np.maximum(dst_roi[:, 1], src_roi[:, 1])
        xmax = np.minimum(dst_roi[:, 0]+dst_roi[:, 2],
                          src_roi[:, 0]+src_roi[:, 2])
        ymax = np.minimum(dst_roi[:, 1]+dst_roi[:, 3],
                          src_roi[:, 1]+src_roi[:, 3])

        interArea = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        boxAArea = dst_roi[:, 2] * dst_roi[:, 3]
        boxBArea = np.tile(src_roi[:, 2] * src_roi[:, 3], (len(dst_roi), 1))
        sumArea = boxAArea + boxBArea

        iou = interArea / (sumArea - interArea + 1e-5)

        return iou

    def _print_name_value(self, name_value, full_arch_name):
        names = name_value.keys()
        values = name_value.values()
        num_values = len(name_value)
        logging.info(
            '| Arch ' +
            ' '.join(['| {}'.format(name) for name in names]) +
            ' |'
        )
        logging.info('|---' * (num_values+1) + '|')
        logging.info(
            '| ' + full_arch_name + ' ' +
            ' '.join(['| {:.3f}'.format(value) for value in values]) +
            ' |'
        )

    def keypoint_eval(self, res_file, res_folder, arch_name='384x288_posefix_on_resnet50', save=False):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_names = ['AP', 'Ap .5', 'AP .75',
                       'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        eval_file = os.path.join(
            res_folder, 'keypoints_%s_results.pkl' % self.image_set)

        if save:
            with open(eval_file, 'wb') as f:
                pickle.dump(coco_eval, f, pickle.HIGHEST_PROTOCOL)
            logging.info('=> coco eval results saved to %s' % eval_file)

        name_values = OrderedDict(info_str)
        if isinstance(name_values, list):
            for name_value in name_values:
                self._print_name_value(name_value, arch_name)
        else:
            self._print_name_value(name_values, arch_name)