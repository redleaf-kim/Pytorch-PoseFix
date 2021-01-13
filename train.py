import json
import argparse
import os
import time
import logging
import random
import numpy as np

import torch
import torch.nn as nn
import torch.autograd.profiler as profiler
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as tfms
from torchvision.transforms.transforms import RandomErasing

from config import config
from config import update_config
from config import update_dir
from config import get_model_name

from model import get_pose_net, get_model_name
from functions import train, valid, evaluate
from utils.utils import save_checkpoint, load_checkpoint, BestLossChecker
from dataset.coco_dataset import COCODataset as coco


logging.basicConfig(filename=config.LOG_DIR, filemode='w',
                    level=logging.INFO, format='%(asctime)s => %(message)s')


class CrossEntropy2d(torch.nn.Module):
    def __init__(self, reduction='none'):
        self.reduction = reduction

    def forward(self, logits, labels, mask):
        B, C = logits.shape[:2]

        logits = -F.log_softmax(logits, dim=-1)
        loss = torch.sum(logits * labels, dim=-1) * mask

        if self.reduction == 'sum':
            loss = torch.sum(loss, dim=-1)
        elif self.reduction == 'mean':
            loss = torch.mean(loss)

        return loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=False,
                        type=str,
                        default=config)

    args, rest = parser.parse_known_args()
    # update config
    # update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)
    parser.add_argument('--checkpoint',
                        help='model checkpoint',
                        type=str)
    parser.add_argument('--train_batch',
                       help='num of batch size',
                       type=int, default=32)
    
    
    # test or validation
    parser.add_argument('--test_batch',
                        help='num of batch for test or validation',
                        type=int, default=32)
    parser.add_argument('--flip_test',
                        help='Filp test usage',
                        type=bool, default=True)
    

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    
    if args.checkpoint:
        config.MODEL.CHECKPOINT = args.checkpoint
        
    config.TRAIN.BATCH_SIZE = args.train_batch
    config.TEST.BATCH_SIZE = args.test_batch
    config.TEST.FLIP_TEST = args.flip_test
    


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
    # model = model.half()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    logging.info(f"Training on CUDA: {torch.cuda.is_available()}")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading process
    train_dataset = coco(
        config,
        config.DATASET.ROOT,
        config.DATASET.TRAIN_SET,
        is_train=True,
        is_eval=False,
        transform=tfms.RandomErasing(p=0.8, scale=(0.5, 0.5)),
    )
    

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    valid_dataset = coco(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        is_train=False,
        is_eval=True,
        transform=None
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        drop_last=False
    )

    start_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = torch.optim.Adam(
        model.parameters(), lr=5e-4, weight_decay=1e-5, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[90, 120], gamma=0.1)
    losschecker = BestLossChecker()

    if config.MODEL.CHECKPOINT is not None:
        info = load_checkpoint(config.MODEL.CHECKPOINT)
        if info is not None:
            start_epoch, model_dic, optim_dic, sched_dic = info

            try:
                model.load_state_dict(model_dic)
                logging.info('Model Loaded.')

                optimizer.load_state_dict(optim_dic)
                logging.info('Optimizer Loaded.')

                if sched_dic is not None:
                    scheduler.load_state_dict(sched_dic)
                else:
                    scheduler.last_epoch = start_epoch
                scheduler.optimizer.load_state_dict(optim_dic)
                logging.info('Scheduler Loaded.')
                logging.info('All Weights Loaded...\n')
            except Exception as e:
                start_epoch = config.TRAIN.BEGIN_EPOCH
                logging.info('Model shape is different. Plz check.')
                logging.info('Starts with init weights...\n')

    end = time.time()
    logging.info('Training Ready\n')

    for epoch in range(start_epoch, config.TRAIN.END_EPOCH):
        if epoch == 10:
            config.TEST.FLIP_TEST = True

        if epoch % 5 == 0 and epoch != 0:
            result = evaluate(config, model, valid_loader)

            if epoch % 100 == 0 and epoch != 0:
                with open(f'{config.result_dir}/data.json', 'w') as f:
                    json.dump(result, f)

                os.makedirs(config.result_dir, exist_ok=True)
            valid_dataset.keypoint_eval(result, config.result_dir + '/pred')
            valid_dataset.keypoint_eval(
                './data/input_pose_path/keypoints_valid2017_results.json', config.result_dir + '/ori/')
            end = time.time()

        losses = train(config, epoch=epoch, loader=train_loader,
                       model=model, optimizer=optimizer)
        total_loss, hm_loss, coord_loss = losses
        is_best = losschecker.update(epoch, total_loss, hm_loss, coord_loss)

        try:
            state_dict = model.module.state_dict()
        except Exception as e:
            state_dict = model.state_dict()

        save_checkpoint({
            'epoch': epoch,
            'model': get_model_name(config),
            'state_dict': state_dict,
            'optimizer': optimizer.state_dict(),
        }, is_best, "./weights_2")

        scheduler.step()
        spent = time.time()-end
        hour = int(spent//3600)
        min = int((spent-hour*3600)//60)
        second = (spent-hour*3600-min*60)

        logging.info(
            f"Epoch {epoch} taken {hour:d}h{min:2d}m {second:2.3f}s\n")
        end = time.time()


if __name__ == "__main__":
    main()
