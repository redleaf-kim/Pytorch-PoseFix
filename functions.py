from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os
import cv2
from nms.nms import oks_nms
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from config import config

logging.basicConfig(filename=config.LOG_DIR, filemode='w',
                    level=logging.INFO, format='%(asctime)s => %(message)s')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def extract_coordinate(config, heatmap_outs, num_joints=17):
    """
        Extracting coordinate from heatmap
        
        [Args]
            [config]: config file
            
            [heatmap_outs]: heatmap result
    """
    
    heatmap_outs = heatmap_outs.detach().cpu()

    batch_size, _, height, width = heatmap_outs.shape
    output_shape = (height, width)

    # coordinate extract from output heatmap
    y = [i for i in range(output_shape[0])]
    x = [i for i in range(output_shape[1])]
    xx, yy = np.meshgrid(x, y)

    xx = torch.from_numpy(xx).float() + 1
    yy = torch.from_numpy(yy).float() + 1

    heatmap_outs = torch.reshape(heatmap_outs, [batch_size, num_joints, -1])
    heatmap_outs = F.softmax(heatmap_outs, dim=-1)
    heatmap_outs = torch.reshape(
        heatmap_outs, [batch_size, num_joints, output_shape[0], output_shape[1]])

    x_out = torch.sum(
        torch.mul(heatmap_outs,
                  torch.reshape(xx, [1, 1, output_shape[0], output_shape[1]]).repeat([batch_size, num_joints, 1, 1])), [2, 3]
    )
    y_out = torch.sum(
        torch.mul(heatmap_outs,
                  torch.reshape(yy, [1, 1, output_shape[0], output_shape[1]]).repeat([batch_size, num_joints, 1, 1])), [2, 3]
    )

    coord_out = torch.cat(
        [torch.reshape(x_out, [batch_size, num_joints, 1]),
         torch.reshape(y_out, [batch_size, num_joints, 1])],
        axis=2
    )
    coord_out = coord_out - 1
    coord_out = coord_out / output_shape[0] * config.MODEL.IMAGE_SIZE[0]
    return coord_out.float()


def render_gaussian_heatmap(config, coord, output_shape, sigma, valid=None):
    """
        Render gaussian heatmap
        
        [Args]
            [config]: config file
            
            [coord]: coords for gaussian rendering
            
            [output_shape]: output heatmap shape
            
            [sigma]: gaussian sigma
    """
    x = [i for i in range(output_shape[1])]
    y = [i for i in range(output_shape[0])]
    xx, yy = np.meshgrid(x, y)

    xx = torch.from_numpy(xx)
    yy = torch.from_numpy(yy)
    xx = torch.reshape(xx.float(), (1, *output_shape, 1))
    yy = torch.reshape(yy.float(), (1, *output_shape, 1))

    x = torch.reshape(coord[:, :, 0], [-1, 1, 1, config.MODEL.NUM_JOINTS]
                      ) / config.MODEL.IMAGE_SIZE[1] * output_shape[1]
    y = torch.reshape(coord[:, :, 1], [-1, 1, 1, config.MODEL.NUM_JOINTS]
                      ) / config.MODEL.IMAGE_SIZE[0] * output_shape[0]

    # TODO 자꾸 여기서 터진다... 뭐가 문제지? -> batch size 128로하면 메모리 overflow로 터진다. ㅠㅠ
    heatmap = torch.exp(-(((xx-x)/torch.tensor(sigma, dtype=torch.float))**2) /
                        2.0 - (((yy-y)/torch.tensor(sigma, dtype=torch.float))**2)/2.0)

    if valid is not None:
        valid_mask = torch.reshape(valid, [-1, 1, 1, config.MODEL.NUM_JOINTS])
        heatmap = heatmap * valid_mask

    return (heatmap * 255.).permute(0, 3, 1, 2)


def render_onehot_heatmap(config, coord, output_shape):
    """
        Render onehot heatmap
        
        [Args]
            [config]: config file
            
            [coord]: coords for gaussian rendering
            
            [output_shape]: output heatmap shape
    """
    
    batch_size = coord.size(0)

    x = torch.reshape(
        coord[:, :, 0] / config.MODEL.IMAGE_SIZE[1] * output_shape[1], [-1])
    y = torch.reshape(
        coord[:, :, 1] / config.MODEL.IMAGE_SIZE[0] * output_shape[0], [-1])
    x_floor = torch.floor(x)
    y_floor = torch.floor(y)

    indices_batch = torch.unsqueeze(torch.arange(batch_size), 0)
    indices_batch = indices_batch.repeat([config.MODEL.NUM_JOINTS, 1])
    indices_batch = torch.transpose(indices_batch, 1, 0).reshape([-1])
    indices_batch = torch.unsqueeze(indices_batch.float(), 1)
    indices_batch = torch.cat(
        [indices_batch, indices_batch, indices_batch, indices_batch], dim=0)

    indices_joint = torch.unsqueeze(torch.arange(
        config.MODEL.NUM_JOINTS).repeat([batch_size]), 1).float()
    indices_joint = torch.cat(
        [indices_joint, indices_joint, indices_joint, indices_joint], dim=0)

    indices_lt = torch.cat(
        [torch.unsqueeze(y_floor, 1), torch.unsqueeze(x_floor, 1)], dim=1)
    indices_lb = torch.cat(
        [torch.unsqueeze(y_floor+1, 1), torch.unsqueeze(x_floor, 1)], dim=1)
    indices_rt = torch.cat(
        [torch.unsqueeze(y_floor, 1), torch.unsqueeze(x_floor+1, 1)], dim=1)
    indices_rb = torch.cat(
        [torch.unsqueeze(y_floor+1, 1), torch.unsqueeze(x_floor+1, 1)], dim=1)

    indices = torch.cat(
        [indices_lt, indices_lb, indices_rt, indices_rb], dim=0)
    indices = torch.cat([indices_batch, indices, indices_joint], dim=1).long()

    prob_lt = (1 - (x - x_floor)) * (1 - (y - y_floor))
    prob_lb = (1 - (x - x_floor)) * (y - y_floor)
    prob_rt = (x - x_floor) * (1 - (y - y_floor))
    prob_rb = (x - x_floor) * (y - y_floor)
    probs = torch.cat([prob_lt, prob_lb, prob_rt, prob_rb], dim=0).float()

    heatmap = torch.zeros(
        (batch_size, config.MODEL.NUM_JOINTS, *output_shape), dtype=torch.float)
    for idx_1, i in enumerate(indices):
        b, y, x, j = i
        if y == 0 and x == 0:
            continue

        if y < output_shape[0] and y >= 0 and x >= 0 and x < output_shape[1]:
            heatmap[b, j, y, x] = probs[idx_1]

    normalizer = torch.reshape(torch.sum(heatmap, dim=[2, 3]), [
                               batch_size, config.MODEL.NUM_JOINTS, 1, 1])
    normalizer = torch.where(
        normalizer == 0, torch.ones_like(normalizer), normalizer)
    heatmap = heatmap / normalizer

    return heatmap


def calc_hm_loss(logits, labels, mask, reduction='none'):
    B, C = logits.shape[:2]

    logits = -F.log_softmax(logits, dim=-1)
    loss = torch.sum(logits * labels, dim=-1) * mask

    if reduction == 'sum':
        loss = torch.sum(loss, dim=-1)
    elif reduction == 'mean':
        loss = torch.mean(loss)

    return loss


def calc_coord_loss(pred, gt, valid_mask):
    return torch.mean(torch.sum(torch.abs(pred-gt) * valid_mask, dim=-1))


def train(config, epoch, loader, model, optimizer, total_epochs=140):
    batch_time = AverageMeter()
    losses = AverageMeter()
    hm_loss = AverageMeter()
    coord_loss = AverageMeter()

    criterion_hm = calc_hm_loss
    criterion_coord = calc_coord_loss

    model.train()

    st = time.time()
    for i, (inputs, erase_inputs, target_coord, input_coord, target_valid, input_pose_valid) in enumerate(loader):
        # measure data loading time
        batch_size = inputs.size(0)

        # compute output
        optimizer.zero_grad()

        input_pose_hms = render_gaussian_heatmap(config, input_coord, config.MODEL.IMAGE_SIZE, config.MODEL.INPUT_SIGMA, input_pose_valid)
        gt_pose_hms = torch.reshape(render_onehot_heatmap(config, target_coord.float(), config.MODEL.OUTPUT_SIZE), [batch_size, config.MODEL.NUM_JOINTS, -1])
        heatmap_outs = model(erase_inputs.cuda().float(),input_pose_hms.cuda().float())
        output = torch.reshape(heatmap_outs, [batch_size, config.MODEL.NUM_JOINTS, -1])

        # calc hm loss
        valid_mask = torch.reshape(target_valid, [batch_size, config.MODEL.NUM_JOINTS])
        loss_hm = criterion_hm(logits=output, labels=gt_pose_hms.cuda().float(), mask=valid_mask.cuda(), reduction='mean')


        # calc coord loss
        pred_coord = extract_coordinate(config, heatmap_outs, config.MODEL.NUM_JOINTS)
        pred_out = pred_coord / config.MODEL.IMAGE_SIZE[0] * config.MODEL.OUTPUT_SIZE[0]
        valid_mask = torch.reshape(target_valid,  [batch_size, config.MODEL.NUM_JOINTS, 1])
        target_out = target_coord / config.MODEL.IMAGE_SIZE[0] * config.MODEL.OUTPUT_SIZE[0]
        loss_coord = criterion_coord(pred_out, target_out, valid_mask)

        loss = loss_hm + loss_coord
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))
        coord_loss.update(loss_coord.item(), inputs.size(0))
        hm_loss.update(loss_hm.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - st)
        st = time.time()


        if i % (len(loader)//10) == 0:
            # visualize coordinates
            vis = True
            if vis:
                random_indicies = torch.randint(0, len(inputs), (5,)).numpy()
                # input heatmaps
                visualize_heatmaps, _ = torch.max(input_pose_hms, dim=1)
                visualize_heatmaps = torch.reshape(visualize_heatmaps, shape=(inputs.shape[0], *config.MODEL.IMAGE_SIZE, 1))

                # pred heatmaps
                visualize_pred_heatmaps, _ = torch.max(heatmap_outs, dim=1)
                visualize_pred_max = torch.max(visualize_pred_heatmaps)
                visualize_pred_min = torch.min(visualize_pred_heatmaps)
                visualize_pred_heatmaps = (visualize_pred_heatmaps-visualize_pred_min)/(visualize_pred_max-visualize_pred_min)
                visualize_pred_heatmaps = torch.reshape(visualize_pred_heatmaps, shape=(inputs.shape[0], 1, *config.MODEL.OUTPUT_SIZE))
                visualize_pred_heatmaps = F.interpolate(visualize_pred_heatmaps, size=config.MODEL.IMAGE_SIZE, mode='bilinear').permute(0, 2, 3, 1)

                # gt heatmaps
                visualize_gt_heatmaps, _ = torch.max(gt_pose_hms, dim=1)
                visualize_gt_heatmaps = torch.reshape(visualize_gt_heatmaps, shape=(inputs.shape[0], 1, *config.MODEL.OUTPUT_SIZE))
                visualize_gt_heatmaps = F.interpolate(visualize_gt_heatmaps, size=config.MODEL.IMAGE_SIZE, mode='bilinear').permute(0, 2, 3, 1)

                for j in random_indicies:
                    # input heatmaps
                    visualize_heatmap = visualize_heatmaps[j].detach().cpu().numpy().astype('uint8')
                    visualize_heatmap = cv2.applyColorMap(visualize_heatmap, cv2.COLORMAP_JET)
                    path = os.path.join(config.train_vis_dir, "heatmap_inputs", str(epoch).zfill(4))
                    os.makedirs(path, exist_ok=True)
                    cv2.imwrite(os.path.join(path, str(i * inputs.shape[0] + j) + '_heatmaps.jpg'), visualize_heatmap)

                    # pred heatmaps
                    visualize_heatmap = visualize_pred_heatmaps[j].detach().cpu().numpy() * 254
                    visualize_heatmap = visualize_heatmap.astype('uint8')
                    visualize_heatmap = cv2.applyColorMap(visualize_heatmap, cv2.COLORMAP_JET)
                    path = os.path.join(config.train_vis_dir, "heatmap_preds", str(epoch).zfill(4))
                    os.makedirs(path, exist_ok=True)
                    cv2.imwrite(os.path.join(path, str(i * inputs.shape[0] + j) + '_heatmaps.jpg'), visualize_heatmap)

                    # gt heatmaps
                    visualize_heatmap = visualize_gt_heatmaps[j].detach().cpu().numpy() * 254
                    visualize_heatmap = visualize_heatmap.astype('uint8')
                    visualize_heatmap = cv2.applyColorMap(visualize_heatmap, cv2.COLORMAP_JET)
                    path = os.path.join(config.train_vis_dir, "heatmap_gts", str(epoch).zfill(4))
                    os.makedirs(path, exist_ok=True)
                    cv2.imwrite(os.path.join(path, str(i * inputs.shape[0] + j) + '_heatmaps.jpg'), visualize_heatmap)

                for j in random_indicies:
                    tmpimg = erase_inputs[j].detach().clone().permute(
                        1, 2, 0).numpy()
                    tmpimg = denormalize_input(config, tmpimg)
                    tmpimg = tmpimg.astype('uint8')
                    tmpkps = np.zeros((3, config.MODEL.NUM_JOINTS))

                    input_valid = torch.reshape(
                        input_pose_valid,  [batch_size, config.MODEL.NUM_JOINTS, 1])
                    tmpkps[:2, :] = (
                        input_coord * input_valid)[j, :, :2].transpose(1, 0)
                    tmpkps[2, :] = input_valid[j, :, 0]
                    _tmpimg = tmpimg.copy()
                    _tmpimg = vis_keypoints(config, _tmpimg, tmpkps)
                    path = os.path.join(
                        config.train_vis_dir, "inputs", str(epoch).zfill(4))
                    os.makedirs(path, exist_ok=True)
                    cv2.imwrite(os.path.join(
                        path, str(i * inputs.shape[0] + j) + '_inputs.jpg'), _tmpimg)

                for j in random_indicies:
                    tmpimg = inputs[j].detach().clone().permute(
                        1, 2, 0).numpy()
                    tmpimg = denormalize_input(config, tmpimg)
                    tmpimg = tmpimg.astype('uint8')
                    tmpkps = np.zeros((3, config.MODEL.NUM_JOINTS))
                    tmpkps[:2, :] = pred_coord[j, :, :2].transpose(1, 0)
                    tmpkps[2, :] = 1
                    _tmpimg = tmpimg.copy()
                    _tmpimg = vis_keypoints(config, _tmpimg, tmpkps)
                    path = os.path.join(
                        config.train_vis_dir, "preds", str(epoch).zfill(4))
                    os.makedirs(path, exist_ok=True)
                    cv2.imwrite(os.path.join(
                        path, str(i * inputs.shape[0] + j) + '_preds.jpg'), _tmpimg)

                for j in random_indicies:
                    tmpimg = inputs[j].detach().clone().permute(
                        1, 2, 0).numpy()
                    tmpimg = denormalize_input(config, tmpimg)
                    tmpimg = tmpimg.astype('uint8')
                    tmpkps = np.zeros((3, config.MODEL.NUM_JOINTS))
                    tmpkps[:2, :] = (target_coord*valid_mask)[j,
                                                              :, :2].transpose(1, 0)
                    tmpkps[2, :] = valid_mask[j, :, 0]
                    _tmpimg = tmpimg.copy()
                    _tmpimg = vis_keypoints(config, _tmpimg, tmpkps)
                    path = os.path.join(
                        config.train_vis_dir, "targets", str(epoch).zfill(4))
                    os.makedirs(path, exist_ok=True)
                    cv2.imwrite(os.path.join(
                        path, str(i * inputs.shape[0] + j) + '_targets.jpg'), _tmpimg)

                del _tmpimg, tmpimg, tmpkps
                del visualize_heatmap
                del visualize_heatmaps
                del visualize_pred_heatmaps

        if i % 100 == 0:
            logging.info(f'Epoch [{str(epoch).zfill(len(str(total_epochs)))} | {str(total_epochs)}] \
->  Batch [{str(i+1).zfill(len(str(len(loader))))} | {str(len(loader))}] \
->  Loss: {losses.avg:.6f} (Heatmap -> {hm_loss.avg:12.6f} | Coordinate -> {coord_loss.avg:12.6f}) \
->  Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) \
->  Speed: {inputs.size(0)/batch_time.val:.1f} samples/s')

    return losses, hm_loss, coord_loss


def vis_keypoints(cfg, img, kps, kp_thresh=0.4, alpha=1):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(cfg.kps_lines) + 2)]
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
            color=colors[len(cfg.kps_lines)], thickness=2, lineType=cv2.LINE_AA)
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder.astype(np.int32)), tuple(
                mid_hip.astype(np.int32)),
            color=colors[len(cfg.kps_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(cfg.kps_lines)):
        i1 = cfg.kps_lines[l][0]
        i2 = cfg.kps_lines[l][1]
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


def normalize_input(cfg, img):
    return img - cfg.pixel_means


def denormalize_input(cfg, img):
    return img + cfg.pixel_means


# TODO flip 적용, evalutation 위해서 json 파일로 저장.
def evaluate(cfg, model, loader):
    cnt = 0
    vis = True
    dump_results = []
    model.eval()

    total_size = len(loader)
    with torch.no_grad():
        for i, (input_value, meta) in enumerate(loader):
            inputs, input_pose_coord, input_pose_valid, input_pose_score, crop_infos = input_value
            if i % (len(loader)//10) == 0:
                logging.info(
                    f'{i/total_size*100:2.2f}%   [{str(i).zfill(len(str(total_size)))} | {total_size}]')

            # compute heatmaps and coords
            input_pose_hms = render_gaussian_heatmap(
                cfg, input_pose_coord, cfg.MODEL.IMAGE_SIZE, cfg.MODEL.INPUT_SIGMA, input_pose_valid)
            heatmap_outs = model(inputs.cuda().float(),
                                 input_pose_hms.cuda().float())
            predicts = extract_coordinate(
                cfg, heatmap_outs, cfg.MODEL.NUM_JOINTS)

            if cfg.TEST.FLIP_TEST:
                flip_imgs = np.flip(inputs.cpu().numpy(), 3).copy()
                flip_imgs = torch.from_numpy(flip_imgs).cuda()
                flip_input_pose_coords = input_pose_coord.clone()
                flip_input_pose_coords[:, :, 0] = cfg.MODEL.IMAGE_SIZE[1] - \
                    1 - flip_input_pose_coords[:, :, 0]
                flip_input_pose_valids = input_pose_valid.clone()
                for (q, w) in cfg.kps_symmetry:
                    flip_input_pose_coords_w, flip_input_pose_coords_q = flip_input_pose_coords[:, w, :].clone(
                    ), flip_input_pose_coords[:, q, :].clone()
                    flip_input_pose_coords[:, q, :], flip_input_pose_coords[:,
                                                                            w, :] = flip_input_pose_coords_w, flip_input_pose_coords_q
                    flip_input_pose_valids_w, flip_input_pose_valids_q = flip_input_pose_valids[:, w].clone(
                    ), flip_input_pose_valids[:, q].clone()
                    flip_input_pose_valids[:, q], flip_input_pose_valids[:,
                                                                         w] = flip_input_pose_valids_w, flip_input_pose_valids_q

                flip_input_pose_hms = render_gaussian_heatmap(
                    cfg, flip_input_pose_coords, cfg.MODEL.IMAGE_SIZE, cfg.MODEL.INPUT_SIGMA, flip_input_pose_valids)
                flip_heatmap_outs = model(
                    flip_imgs.cuda().float(), flip_input_pose_hms.cuda().float())
                flip_coords = extract_coordinate(
                    cfg, flip_heatmap_outs.float(), cfg.MODEL.NUM_JOINTS)

                flip_coords[:, :, 0] = cfg.MODEL.IMAGE_SIZE[1] - \
                    1 - flip_coords[:, :, 0]
                for (q, w) in cfg.kps_symmetry:
                    flip_coord_w, flip_coord_q = flip_coords[:, w, :].clone(
                    ), flip_coords[:, q, :].clone()
                    flip_coords[:, q, :], flip_coords[:,
                                                      w, :] = flip_coord_w, flip_coord_q

                predicts += flip_coords
                predicts /= 2

            kps_result = np.zeros((len(inputs), cfg.MODEL.NUM_JOINTS, 3))
            area_save = np.zeros(len(inputs))

            
            for j in range(len(predicts)):
                kps_result[j, :, :2] = predicts[j]
                kps_result[j, :, 2] = input_pose_score[j]

                crop_info = crop_infos[j, :]
            #     area = (crop_info[2] - crop_info[0]) * \
            #         (crop_info[3] - crop_info[1])

            #     # if vis and np.any(kps_result[j,:,2]) > 0.9 and area > 96**2:
            #     #     tmpimg = inputs[j].detach().clone().permute(1, 2, 0).numpy()
            #     #     tmpimg = denormalize_input(cfg, tmpimg)
            #     #     tmpimg = tmpimg.astype('uint8')
            #     #     tmpkps = np.zeros((3,cfg.MODEL.NUM_JOINTS))
            #     #     tmpkps[:2,:] = kps_result[j,:,:2].transpose(1,0)
            #     #     tmpkps[2,:] = kps_result[j,:,2]
            #     #     _tmpimg = tmpimg.copy()
            #     #     _tmpimg = vis_keypoints(cfg, _tmpimg, tmpkps)
            #     #     path = os.path.join(cfg.valid_vis_dir, "flip_output", str('evaluate').zfill(4))
            #     #     os.makedirs(path, exist_ok=True)
            #     #     cv2.imwrite(os.path.join(path, str(i * inputs.shape[0] + j) + '_output.jpg'), _tmpimg)

                # map back to original images
                for k in range(cfg.MODEL.NUM_JOINTS):
                    kps_result[j, k, 0] = kps_result[j, k, 0] / cfg.MODEL.IMAGE_SIZE[1] * (
                        crop_infos[j][2] - crop_infos[j][0]) + crop_infos[j][0]
                    kps_result[j, k, 1] = kps_result[j, k, 1] / cfg.MODEL.IMAGE_SIZE[0] * (
                        crop_infos[j][3] - crop_infos[j][1]) + crop_infos[j][1]

            #     area_save[j] = (crop_infos[j][2] - crop_infos[j]
            #                     [0]) * (crop_infos[j][3] - crop_infos[j][1])

            # # folder = 'valid'
            # # if vis and i % 100 == 0:
            # #     for j in range(len(predicts)):
            # #         if np.any(kps_result[j,:,2] > 0.9):
            # #             path = os.path.join(cfg.DATASET.ROOT, 'images', 'valid', meta['imgpath'][j])
            # #             img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            # #             tmpimg = cv2.imread(os.path.join(cfg.DATASET.ROOT, 'images', folder,meta['imgpath'][j]))
            # #             tmpimg = tmpimg.astype('uint8')

            # #             tmpkps_pred = np.zeros((3,cfg.MODEL.NUM_JOINTS))
            # #             tmpkps_pred[:2,:] = kps_result[j, :, :2].transpose(1,0)
            # #             tmpkps_pred[2,:] = kps_result[j, :, 2]

            # #             tmpkps_orig = np.zeros((3,cfg.MODEL.NUM_JOINTS))
            # #             tmpkps_orig[:2,:] = kps_result[j, :, :2].transpose(1,0)
            # #             tmpkps_orig[2,:] = kps_result[j, :, 2]

            # #             tmpimg_pred = vis_keypoints(cfg, tmpimg, tmpkps_pred, kp_thresh=0.1)
            # #             tmpimg_orig = vis_keypoints(cfg, tmpimg, tmpkps_orig, kp_thresh=0.1)

            # #             path_orig = os.path.join(cfg.valid_vis_dir, "flip_output", str('evaluate_orig'))
            # #             path_pred = os.path.join(cfg.valid_vis_dir, "flip_output", str('evaluate_pred'))
            # #             os.makedirs(path_orig, exist_ok=True)
            # #             os.makedirs(path_pred, exist_ok=True)

            # #             cv2.imwrite(os.path.join(path_orig, str(cnt) + '.jpg'), tmpimg_orig)
            # #             cv2.imwrite(os.path.join(path_pred, str(cnt) + '.jpg'), tmpimg_pred)
            # #             cnt += 1

            # oks nms
            if cfg.oks_nms:
                nms_kps = np.delete(kps_result, cfg.ignore_kps, 1)
                nms_score = np.mean(nms_kps[:, :, 2], axis=1)
                nms_kps[:, :, 2] = 1
                nms_kps = nms_kps.reshape(len(kps_result), -1)
                nms_sigmas = np.delete(cfg.kps_sigmas, cfg.ignore_kps)
                keep = oks_nms(nms_kps, nms_score, area_save,
                               cfg.oks_nms_thr, nms_sigmas)
                if len(keep) > 0:
                    kps_result = kps_result[keep, :, :]
                    area_save = area_save[keep]
                    meta['image_id'] = meta['image_id'].numpy()[keep]
                    meta['center'][0] = meta['center'][0].numpy()[keep]
                    meta['center'][1] = meta['center'][1].numpy()[keep]
                    meta['scale'][0] = meta['scale'][0].numpy()[keep]
                    meta['scale'][1] = meta['scale'][1].numpy()[keep]

            score_result = np.copy(kps_result[:, :, 2])
            kps_result[:, :, 2] = 1
            kps_result = kps_result.reshape(-1, cfg.MODEL.NUM_JOINTS*3)

            # save result
            for j in range(len(kps_result)):
                result = dict(image_id=int(meta['image_id'][j]), category_id=1, score=float(round(np.mean(score_result[j]), 4)),
                              keypoints=kps_result[j].round(3).tolist(),
                              center=[meta['center'][0][j],
                                      meta['center'][1][j]],
                              scale=[meta['scale'][0][j], meta['scale'][1][j]])

                dump_results.append(result)

    return dump_results


def valid(config, epoch, loader, model, total_epochs=140):
    model.eval()
    batch_time = AverageMeter()

    st = time.time()
    with torch.no_grad():
        for i, (inputs, input_pose_coord, input_pose_valid, input_pose_score, crop_info) in enumerate(loader):
            # compute heatmaps and coords
            input_pose_hms = render_gaussian_heatmap(
                config, input_pose_coord, config.MODEL.IMAGE_SIZE, config.MODEL.INPUT_SIGMA, input_pose_valid)
            heatmap_outs = model(inputs.cuda().float(),
                                 input_pose_hms.cuda().float())
            predicts = extract_coordinate(
                config, heatmap_outs, config.MODEL.NUM_JOINTS)

            # compute flip image's heatmaps and coords
            if config.TEST.FLIP_TEST:
                flip_imgs = np.flip(inputs.cpu().numpy(), 3).copy()
                flip_imgs = torch.from_numpy(flip_imgs).cuda()
                flip_input_pose_coords = input_pose_coord.clone()
                flip_input_pose_coords[:, :, 0] = config.MODEL.IMAGE_SIZE[1] - \
                    1 - flip_input_pose_coords[:, :, 0]
                flip_input_pose_valids = input_pose_valid.clone()
                for (q, w) in config.kps_symmetry:
                    flip_input_pose_coords_w, flip_input_pose_coords_q = flip_input_pose_coords[:, w, :].clone(
                    ), flip_input_pose_coords[:, q, :].clone()
                    flip_input_pose_coords[:, q, :], flip_input_pose_coords[:,
                                                                            w, :] = flip_input_pose_coords_w, flip_input_pose_coords_q
                    flip_input_pose_valids_w, flip_input_pose_valids_q = flip_input_pose_valids[:, w].clone(
                    ), flip_input_pose_valids[:, q].clone()
                    flip_input_pose_valids[:, q], flip_input_pose_valids[:,
                                                                         w] = flip_input_pose_valids_w, flip_input_pose_valids_q

                flip_input_pose_hms = render_gaussian_heatmap(
                    config, flip_input_pose_coords, config.MODEL.IMAGE_SIZE, config.MODEL.INPUT_SIGMA, flip_input_pose_valids)
                flip_heatmap_outs = model(
                    flip_imgs.cuda().float(), flip_input_pose_hms.cuda().float())
                flip_coords = extract_coordinate(
                    config, flip_heatmap_outs.float(), config.MODEL.NUM_JOINTS)

                flip_coords[:, :, 0] = config.MODEL.IMAGE_SIZE[1] - \
                    1 - flip_coords[:, :, 0]
                for (q, w) in config.kps_symmetry:
                    flip_coord_w, flip_coord_q = flip_coords[:, w, :].clone(
                    ), flip_coords[:, q, :].clone()
                    flip_coords[:, q, :], flip_coords[:,
                                                      w, :] = flip_coord_w, flip_coord_q

                predicts += flip_coords
                predicts /= 2

            # calc inference fps
            batch_time.update(time.time() - st)
            st = time.time()

            # visualize
            vis = True
            if vis and i % 10 == 0:
                random_indicies = torch.randint(0, len(inputs), (5,)).numpy()
                visualize_heatmaps, _ = torch.max(input_pose_hms, dim=1)
                visualize_heatmaps = torch.reshape(visualize_heatmaps, shape=(
                    inputs.shape[0], *config.MODEL.IMAGE_SIZE, 1))
                visualize_pred_heatmaps, _ = torch.max(heatmap_outs, dim=1)
                visualize_pred_max = torch.max(visualize_pred_heatmaps)
                visualize_pred_min = torch.min(visualize_pred_heatmaps)
                visualize_pred_heatmaps = (
                    visualize_pred_heatmaps-visualize_pred_min)/(visualize_pred_max-visualize_pred_min)
                visualize_pred_heatmaps = torch.reshape(visualize_pred_heatmaps, shape=(
                    inputs.shape[0], 1, *config.MODEL.OUTPUT_SIZE, ))
                visualize_pred_heatmaps = F.interpolate(
                    visualize_pred_heatmaps, size=config.MODEL.IMAGE_SIZE, mode='bilinear').permute(0, 2, 3, 1)

                for j in random_indicies:
                    visualize_heatmap = visualize_heatmaps[j].detach(
                    ).cpu().numpy().astype('uint8')
                    visualize_heatmap = cv2.applyColorMap(
                        visualize_heatmap, cv2.COLORMAP_JET)

                    path = os.path.join(
                        config.valid_vis_dir, "heatmap_inputs", str(epoch).zfill(4))
                    os.makedirs(path, exist_ok=True)
                    cv2.imwrite(os.path.join(
                        path, str(i * inputs.shape[0] + j) + '_heatmaps.jpg'), visualize_heatmap)

                    visualize_heatmap = visualize_pred_heatmaps[j].detach(
                    ).cpu().numpy() * 254
                    visualize_heatmap = visualize_heatmap.astype('uint8')
                    visualize_heatmap = cv2.applyColorMap(
                        visualize_heatmap, cv2.COLORMAP_JET)
                    path = os.path.join(
                        config.valid_vis_dir, "heatmap_preds", str(epoch).zfill(4))
                    os.makedirs(path, exist_ok=True)
                    cv2.imwrite(os.path.join(
                        path, str(i * inputs.shape[0] + j) + '_heatmaps.jpg'), visualize_heatmap)

                for j in random_indicies:
                    tmpimg = inputs[j].detach().clone(
                    ).cpu().permute(1, 2, 0).numpy()
                    tmpimg = denormalize_input(config, tmpimg)
                    tmpimg = tmpimg.astype('uint8')
                    tmpkps = np.zeros((3, config.MODEL.NUM_JOINTS))
                    tmpkps[:2, :] = input_pose_coord[j, :, :2].transpose(1, 0)
                    tmpkps[2, :] = input_pose_score[j]
                    _tmpimg = tmpimg.copy()
                    _tmpimg = vis_keypoints(config, _tmpimg, tmpkps)
                    path = os.path.join(
                        config.valid_vis_dir, "inputs", str(epoch).zfill(4))
                    os.makedirs(path, exist_ok=True)
                    cv2.imwrite(os.path.join(
                        path, str(i * inputs.shape[0] + j) + '_inputs.jpg'), _tmpimg)

                for j in random_indicies:
                    tmpimg = inputs[j].detach().clone(
                    ).cpu().permute(1, 2, 0).numpy()
                    tmpimg = denormalize_input(config, tmpimg)
                    tmpimg = tmpimg.astype('uint8')
                    tmpkps = np.zeros((3, config.MODEL.NUM_JOINTS))
                    tmpkps[:2, :] = predicts[j, :, :2].transpose(1, 0)
                    # tmpkps[2,:] = input_pose_score[j]
                    tmpkps[2, :] = 1
                    _tmpimg = tmpimg.copy()
                    _tmpimg = vis_keypoints(config, _tmpimg, tmpkps)
                    path = os.path.join(
                        config.valid_vis_dir, "preds", str(epoch).zfill(4))
                    os.makedirs(path, exist_ok=True)
                    cv2.imwrite(os.path.join(
                        path, str(i*inputs.shape[0] + j) + '_preds.jpg'), _tmpimg)

                # if config.TEST.FLIP_TEST:
                #     for j in random_indicies:
                #         tmpimg = inputs[j].detach().clone().cpu().permute(1, 2, 0).numpy()
                #         tmpimg = denormalize_input(config, tmpimg)
                #         tmpimg = tmpimg.astype('uint8')
                #         tmpkps = np.zeros((3,config.MODEL.NUM_JOINTS))
                #         tmpkps[:2,:] = flip_coords[j,:,:2].transpose(1,0)
                #         tmpkps[2,:] = input_pose_score[j]
                #         _tmpimg = tmpimg.copy()
                #         _tmpimg = vis_keypoints(config, _tmpimg, tmpkps)
                #         path = os.path.join(config.valid_vis_dir, "flip_preds", str(epoch).zfill(4))
                #         os.makedirs(path, exist_ok=True)
                #         cv2.imwrite(os.path.join(path, str(i*inputs.shape[0] + j) + '_preds.jpg'), _tmpimg)

                del _tmpimg, tmpimg, tmpkps
                del visualize_heatmap
                del visualize_heatmaps
                del visualize_pred_heatmaps

                logging.info(f'Epoch [{str(epoch).zfill(len(str(total_epochs)))} | {str(total_epochs)}] \
->  Batch [{str(i+1).zfill(len(str(len(loader))))} | {str(len(loader))}] \
->  Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) \
->  Speed: {inputs.size(0)/batch_time.val:.1f} samples/s')

        logging.info(f'Validation on Epoch {epoch} has done.\n')
