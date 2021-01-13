import argparse
import os, time
import logging
import random
import json, cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.autograd.profiler as profiler
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as tfms

from config import config
from config import update_config
from config import update_dir
from config import get_model_name

from scipy.io import savemat
from model import get_pose_net, get_model_name
from functions import evaluate
from utils.utils import save_checkpoint, load_checkpoint, BestLossChecker
from dataset.coco_dataset import COCODataset as coco
from utils.transforms import get_affine_transform, affine_transform

from torch.utils.data import Dataset, DataLoader
from functions import extract_coordinate, render_gaussian_heatmap, render_onehot_heatmap, vis_keypoints

logging.basicConfig(filename=f"../loggers/evaluation_Logger.log", filemode='w', level=logging.INFO, format='%(asctime)s => %(message)s')


class TestDatset(Dataset):
    def __init__(self, cfg, video_dir, annot_dir, save_dir, frame_area=(600, 150)):
        super(TestDatset, self).__init__()
        
        self.cfg = cfg
        self.video_dir = video_dir
        self.annot_dir = annot_dir
        self.save_dir  = save_dir
        self.frame_area = frame_area
        
        self.cap = cv2.VideoCapture(video_dir)
        self.data = self.load_data()
                
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int):
        data = self.data[idx]
        frame_idx = data["image_id"]
        x,y,w,h = data['bbox']
        # x1,y1,x2,y2 = data['orig_bbox']
        
        
        self.cap.set(1, frame_idx)
        _, img = self.cap.read()
        
        aspect_ratio = self.cfg.MODEL.IMAGE_SIZE[1] / self.cfg.MODEL.IMAGE_SIZE[0]
        centre = np.array([x+w*.5, y+h*.5])

        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array([w, h]) * 1.25
        rotation = 0
        
        trans = get_affine_transform(centre, scale, rotation, (self.cfg.MODEL.IMAGE_SIZE[1], self.cfg.MODEL.IMAGE_SIZE[0]))
        cropped_img = cv2.warpAffine(img, trans, (self.cfg.MODEL.IMAGE_SIZE[1], self.cfg.MODEL.IMAGE_SIZE[0]), flags=cv2.INTER_LINEAR)
        cropped_img = normalize_input(cropped_img, self.cfg)
        
        # cv2.imshow("orig", img)
        # cropped_show = denormalize_input(cropped_img, self.cfg).copy().astype(np.uint8)
        # cv2.imshow("crop", cropped_show)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        
        estimated_joints = np.zeros((self.cfg.MODEL.NUM_JOINTS, 3), dtype=np.float)
        offsets = np.zeros((self.cfg.MODEL.NUM_JOINTS, 2), dtype=np.float)
        offsets[:, 0] = self.frame_area[0]
        offsets[:, 1] = self.frame_area[1]
        
        estimated_joints[:, :2] = np.array(data['joints']).reshape(self.cfg.MODEL.NUM_JOINTS, 2)
        estimated_joints[:, :2] += offsets
        estimated_joints[:,  2] = np.array(data['score'])
        
        for j in range(self.cfg.MODEL.NUM_JOINTS):
            if estimated_joints[j,2] > 0:
                estimated_joints[j,:2] = affine_transform(estimated_joints[j,:2], trans)
                estimated_joints[j, 2] *= ((estimated_joints[j,0] >= 0) & (estimated_joints[j,0] < self.cfg.MODEL.IMAGE_SIZE[1]) & (estimated_joints[j,1] >= 0) & (estimated_joints[j,1] < self.cfg.MODEL.IMAGE_SIZE[0]))

        input_pose_coord = estimated_joints[:,:2]
        input_pose_valid = np.array([1 if i not in self.cfg.ignore_kps else 0 for i in range(self.cfg.MODEL.NUM_JOINTS)])
        input_pose_score = estimated_joints[:, 2]
        
        crop_info = np.asarray([centre[0]-scale[0]*0.5, centre[1]-scale[1]*0.5, centre[0]+scale[0]*0.5, centre[1]+scale[1]*0.5])
        
        
        return [torch.from_numpy(cropped_img).float().permute(2, 0, 1), 
                input_pose_coord,
                input_pose_valid, 
                input_pose_score,
                crop_info,
                frame_idx,
        ]


    def load_data(self):
        save_dir  = self.save_dir
        video_dir = self.video_dir
        annot_dir = self.annot_dir
        
        with open(annot_dir, 'r') as f:
            orig_data = json.load(f)
        
        input_pose = []
        for i, row_datas in enumerate(orig_data.items()):
            frame_id = i
            
            for data in row_datas[1]['person_joints']:
                orig_bbox = data['bbox']
                joints = data['joints']
                score = data['score']
                
                # x1, y1, x2, y2 = orig_bbox
                x = np.array(joints[::2])
                y = np.array(joints[1::2])
                x1, x2, y1, y2  = np.min(x), np.max(x), np.min(y), np.max(y)
                
                x1, x2 = x1+self.frame_area[0], x2+self.frame_area[0]
                y1, y2 = y1+self.frame_area[1], y2+self.frame_area[1]
                
                width = (x2 - x1)
                center = (x1 + x2)/2.
                xmin = center - width/2. * 1.2
                xmax = center + width/2. * 1.2

                
                height = (y2 - y1)
                center = (y2 + y1)/2.
                ymin = center - height/2. * 2
                ymax = center + height/2. * 2
                bbox = [xmin,ymin,xmax-xmin,ymax-ymin]

                
            
                data = dict(image_id=frame_id, bbox=bbox, score=score, joints=joints, orig_bbox=orig_bbox)
                input_pose.append(data)
        print("Original Annotation Ready.\n")
        return input_pose
    
    
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=False,
                        type=str,
                        default=config)
    
    parser.add_argument('--checkpoint',
                        required=True,
                        help='model checkpoint',
                        type=str)
    
    # test or validation
    parser.add_argument('--test_batch',
                        help='num of batch for test or validation',
                        type=int, default=32)
    parser.add_argument('--flip_test',
                        help='Filp test usage',
                        type=bool, default=True)
    
    
    parser.add_argument('--video_path',
                        required=True,
                        help='test video path',
                        type=str)
    parser.add_argument('--detection_json',
                        required=True,
                        help='original pose estimation result which wanna try to fix',
                        type=str)
    

    args, rest = parser.parse_known_args()
    args = parser.parse_args()

    return args


def reset_config(config, args):
    config.MODEL.CHECKPOINT = args.checkpoint
    config.TEST.BATCH_SIZE = args.test_batch
    config.TEST.FLIP_TEST = args.flip_test


def normalize_input(img, cfg):
    return img - cfg.pixel_means


def denormalize_input(img, cfg):
    return img + cfg.pixel_means


        
def main():
    # for reproduciblity
    random_seed = 2020
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    
    
    args = parse_args()
    reset_config(config, args)
    
    # model loading
    model = get_pose_net(
        config, is_train=True
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Model on CUDA: {torch.cuda.is_available()}")
    
    if config.MODEL.CHECKPOINT is not None:
        info = load_checkpoint(config.MODEL.CHECKPOINT)
        if info is not None:
            _, model_dic, _, _ = info
            
            try:
                model.load_state_dict(model_dic)
                logging.info('Model Loaded.\n')
            except Exception as e:
                raise FileNotFoundError('Model shape is different. Plz check.')
                
    
    # dataset = TestDatset(config, './test/cam3_test_short.mp4', './test/detection_result_cam3_test.json', None, (0, 0))
    dataset = TestDatset(config, args.video_path, args.detection_json, None, (0, 0))
    loader  = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        drop_last=False,
    )
    
    cnt = 0
    vis = True
    total_size = len(loader)
    
    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader)):
            if i == 1000:
                break
            
            if i % (len(loader)//10) == 0:
                logging.info(f'{i/total_size*100:2.2f}%   [{str(i).zfill(len(str(total_size)))} | {total_size}]')
            
            imgs, coords, valids, scores, crop_infos, frame_ids = data
            
            input_pose_hms = render_gaussian_heatmap(config, coords, config.MODEL.IMAGE_SIZE, config.MODEL.INPUT_SIGMA, valids)
            heatmap_outs = model(imgs.cuda().float(), input_pose_hms.cuda().float())
            predicts = extract_coordinate(config, heatmap_outs, config.MODEL.NUM_JOINTS)
            
            if config.TEST.FLIP_TEST:
                flip_imgs = np.flip(imgs.cpu().numpy(), 3).copy()
                flip_imgs = torch.from_numpy(flip_imgs).cuda()
                flip_input_pose_coords = coords.clone()
                flip_input_pose_coords[:,:,0] = config.MODEL.IMAGE_SIZE[1] - 1 - flip_input_pose_coords[:,:,0]
                flip_input_pose_valids = valids.clone()
                for (q, w) in config.kps_symmetry:
                    flip_input_pose_coords_w, flip_input_pose_coords_q = flip_input_pose_coords[:,w,:].clone(), flip_input_pose_coords[:,q,:].clone()
                    flip_input_pose_coords[:,q,:], flip_input_pose_coords[:,w,:] = flip_input_pose_coords_w, flip_input_pose_coords_q
                    flip_input_pose_valids_w, flip_input_pose_valids_q = flip_input_pose_valids[:,w].clone(), flip_input_pose_valids[:,q].clone()
                    flip_input_pose_valids[:,q], flip_input_pose_valids[:,w] = flip_input_pose_valids_w, flip_input_pose_valids_q

                flip_input_pose_hms = render_gaussian_heatmap(config, flip_input_pose_coords, config.MODEL.IMAGE_SIZE, config.MODEL.INPUT_SIGMA, flip_input_pose_valids)
                flip_heatmap_outs = model(flip_imgs.cuda().float(), flip_input_pose_hms.cuda().float())
                flip_coords = extract_coordinate(config, flip_heatmap_outs.float(), config.MODEL.NUM_JOINTS)

                flip_coords[:,:,0] = config.MODEL.IMAGE_SIZE[1] - 1 - flip_coords[:,:,0]
                for (q, w) in config.kps_symmetry:
                    flip_coord_w, flip_coord_q = flip_coords[:,w,:].clone(), flip_coords[:,q,:].clone()
                    flip_coords[:,q,:], flip_coords[:,w,:] = flip_coord_w, flip_coord_q
                
                predicts += flip_coords
                predicts /= 2

            
            kps_result = np.zeros((len(imgs), config.MODEL.NUM_JOINTS, 3))
            area_save = np.zeros(len(imgs))
            
            visualize_pred_heatmaps, _ = torch.max(heatmap_outs, dim=1)
            visualize_pred_max = torch.max(visualize_pred_heatmaps)
            visualize_pred_min = torch.min(visualize_pred_heatmaps)
            visualize_pred_heatmaps = (visualize_pred_heatmaps-visualize_pred_min)/(visualize_pred_max-visualize_pred_min)
            visualize_pred_heatmaps = torch.reshape(visualize_pred_heatmaps, shape=(imgs.shape[0], 1, *config.MODEL.OUTPUT_SIZE))
            visualize_pred_heatmaps = torch.nn.functional.interpolate(visualize_pred_heatmaps, size=config.MODEL.IMAGE_SIZE, mode='bilinear').permute(0, 2, 3, 1)
            
            
            for j in range(len(predicts)):
                visualize_pred_heatmap = visualize_pred_heatmaps[j].detach().cpu().numpy() * 254
                visualize_pred_heatmap = visualize_pred_heatmap.astype('uint8')
                visualize_pred_heatmap = cv2.applyColorMap(visualize_pred_heatmap, cv2.COLORMAP_JET)
                
                kps_result[j, :, :2] = predicts[j]
                kps_result[j, :, 2]  = valids[j]
                

                crop_info = crop_infos[j, :]
                area = (crop_info[2] - crop_info[0]) * (crop_info[3] - crop_info[1])
                
                if vis and np.any(kps_result[j,:,2]) > 0.9 and area > 96**2:
                    tmpimg = imgs[j].detach().clone().permute(1, 2, 0).numpy()
                    tmpimg = denormalize_input(tmpimg, config)
                    tmpimg = tmpimg.astype('uint8')
                    tmpkps = np.zeros((3,config.MODEL.NUM_JOINTS))
                    tmpkps[:2,:] = kps_result[j,:,:2].transpose(1,0)
                    tmpkps[2,:] = kps_result[j,:,2]
                    _tmpimg = tmpimg.copy()
                    _tmpimg = vis_keypoints(config, _tmpimg, tmpkps)
                    
                    alpha = 0.4
                    _tmpimg = cv2.addWeighted(
                        _tmpimg,
                        1.0 - alpha,
                        visualize_pred_heatmap,
                        alpha,
                        0
                    )
                    
                    tmpkps = np.zeros((3,config.MODEL.NUM_JOINTS))
                    tmpkps[:2,:] = coords[j,:,:2].transpose(1,0)
                    tmpkps[2,:] = 1
                    
                    _tmpimg_orig = tmpimg.copy()
                    
                    # _tmpimg_orig = cv2.addWeighted(
                    #     _tmpimg_orig,
                    #     1.0 - alpha,
                    #     input_pose_hms[j].cpu().numpy(),
                    #     alpha,
                    #     0
                    # )
                    
                    _tmpimg_orig = vis_keypoints(config, _tmpimg_orig, tmpkps)
                    
                    
                    path = os.path.join('./test_result', str('cropped_pred').zfill(4))
                    os.makedirs(path, exist_ok=True)
                    cv2.imwrite(os.path.join(path, str(i * imgs.shape[0] + j) + '_output.jpg'), _tmpimg)
                    
                    path = os.path.join('./test_result', str('cropped_orig').zfill(4))
                    os.makedirs(path, exist_ok=True)
                    cv2.imwrite(os.path.join(path, str(i * imgs.shape[0] + j) + '_output.jpg'), _tmpimg_orig)


                for k in range(config.MODEL.NUM_JOINTS):
                    kps_result[j, k, 0] = kps_result[j, k, 0] / config.MODEL.IMAGE_SIZE[1] * (\
                    crop_infos[j][2] - crop_infos[j][0]) + crop_infos[j][0]
                    kps_result[j, k, 1] = kps_result[j, k, 1] / config.MODEL.IMAGE_SIZE[0] * (\
                    crop_infos[j][3] - crop_infos[j][1]) + crop_infos[j][1]
                    
                    # for mapping back to original
                    coords[j, k, 0] = coords[j, k, 0] / config.MODEL.IMAGE_SIZE[1] * (\
                    crop_infos[j][2] - crop_infos[j][0]) + crop_infos[j][0]
                    coords[j, k, 1] = coords[j, k, 1] / config.MODEL.IMAGE_SIZE[0] * (\
                    crop_infos[j][3] - crop_infos[j][1]) + crop_infos[j][1]
                
                area_save[j] = (crop_infos[j][2] - crop_infos[j][0]) * (crop_infos[j][3] - crop_infos[j][1])
            
            
            if vis:
                visualize_pred_heatmaps, _ = torch.max(heatmap_outs, dim=1)
                visualize_pred_max = torch.max(visualize_pred_heatmaps)
                visualize_pred_min = torch.min(visualize_pred_heatmaps)
                visualize_pred_heatmaps = (visualize_pred_heatmaps-visualize_pred_min)/(visualize_pred_max-visualize_pred_min)
                visualize_pred_heatmaps = torch.reshape(visualize_pred_heatmaps, shape=(imgs.shape[0], 1, *config.MODEL.OUTPUT_SIZE))
                
                for j in range(len(predicts)):
                    if np.any(kps_result[j,:,2] > 0.9):
                        dataset.cap.set(1, int(frame_ids[j].data))
                        _, tmpimg = dataset.cap.read()
                        tmpimg = tmpimg.astype('uint8')
                        
                        tmpkps_pred = np.zeros((3, config.MODEL.NUM_JOINTS))
                        tmpkps_pred[:2,:] = kps_result[j, :, :2].transpose(1,0)
                        tmpkps_pred[2,:] = kps_result[j, :, 2]
                        
                        tmpkps_orig = np.zeros((3, config.MODEL.NUM_JOINTS))
                        tmpkps_orig[:2,:] = coords[j, :, :2].transpose(1,0)
                        tmpkps_orig[2,:] = scores[j]
                        
                        tmpimg_pred = vis_keypoints(config, tmpimg, tmpkps_pred, kp_thresh=0.1)
                        

                        # x1,y1,x2,y2 = crop_infos[j]
                        # h = int(min(y2, tmpimg_pred.shape[0])-int(y1))
                        # w = int(min(x2, tmpimg_pred.shape[1])-int(x1))
                        
                        # visualize_pred_heatmap = torch.nn.functional.interpolate(visualize_pred_heatmaps[j].unsqueeze(0), 
                        #                                                     size=(h, w), 
                        #                                                     mode='bilinear').permute(0, 2, 3, 1).detach().cpu().numpy() * 254
                        # visualize_pred_heatmap = visualize_pred_heatmap[0].astype('uint8')
                        # visualize_pred_heatmap = cv2.applyColorMap(visualize_pred_heatmap, cv2.COLORMAP_JET)
                        
                        
                        # alpha = 0.4
                        # tmpimg_pred[int(y1):int(y1+h), int(x1):int(x1+w)] = cv2.addWeighted(
                        #     tmpimg_pred[int(y1):int(y2), int(x1):int(x2)], 
                        #     1.0 - alpha, 
                        #     visualize_pred_heatmap, 
                        #     alpha, 0)
                        
                        # cv2.imshow("all", tmpimg_pred)
                        # cv2.waitKey()
                        # cv2.destroyWindow("all")
                        
                        tmpimg_orig = vis_keypoints(config, tmpimg, tmpkps_orig, kp_thresh=0.1)
                        
                        
                        path_orig = os.path.join('test_result', str('evaluate_orig'))
                        path_pred = os.path.join('test_result', str('evaluate_pred'))
                        os.makedirs(path_orig, exist_ok=True)
                        os.makedirs(path_pred, exist_ok=True)
                        
                        cv2.imwrite(os.path.join(path_orig, str(cnt) + '.jpg'), tmpimg_orig)
                        cv2.imwrite(os.path.join(path_pred, str(cnt) + '.jpg'), tmpimg_pred)
                        cnt += 1
    
                
    end = time.time()
    logging.info('Test Start\n')
    dataset.cap.relase()
            
            
if __name__ == "__main__":
    main()