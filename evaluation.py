import argparse
import os, time
import logging
import random
import json
import numpy as np

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


logging.basicConfig(filename=f"./loggers/evaluation_Logger.log", filemode='w', level=logging.INFO, format='%(asctime)s => %(message)s')
    
    
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=False,
                        type=str,
                        default=config)

    args, rest = parser.parse_known_args()
    args = parser.parse_args()

    return args

        
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
    
    
    # model loading
    model = get_pose_net(
        config, is_train=True
    )
    # model = model.half()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    valid_dataset = coco(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        is_train=False,
        is_eval=True,
        transform=tfms.Compose([
            tfms.ToTensor(),
        ])
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        # num_workers=confi g.WORKERS,
        # pin_memory=True
        drop_last=False,
    )
            

    if config.MODEL.CHECKPOINT is not None:
        info = load_checkpoint(config.MODEL.CHECKPOINT)
        if info is not None:
            _, model_dic, _, _ = info
            
            try:
                model.load_state_dict(model_dic)
                logging.info('Model Loaded.\n')
            except Exception as e:
                raise FileNotFoundError('Model shape is different. Plz check.')
            

    end = time.time()
    logging.info('Evaluation Ready\n')
    
    result = evaluate(config, model, valid_loader)
    
    with open(f'{config.result_dir}/data.json', 'w') as f:
        json.dump(result, f)  
    logging.info(f"Taken {time.time()-end:.5f}s\n")
    
    
    os.makedirs(config.result_dir, exist_ok=True)
    
    logging.info(f"From a Pose estimator.\n")
    valid_dataset.keypoint_eval('/home/mah/workspace/PoseFix/data/input_pose_path/keypoints_valid2017_results.json', config.result_dir + '/ori/')
    
    logging.info(f"Pose Estimator with PoseFix.\n")
    valid_dataset.keypoint_eval(result, config.result_dir + '/pred')
    
            
            
if __name__ == "__main__":
    main()