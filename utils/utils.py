import os
import torch
import logging

from config import config

logging.basicConfig(filename=config.LOG_DIR, filemode='w',
                    level=logging.INFO, format='%(asctime)s => %(message)s')
# logging.basicConfig(level=logging.INFO, format='%(name)s :: %(asctime)s => %(message)s')

class BestLossChecker():
    def __init__(self, value=None):
        self.best = value
        self.hm_loss = None
        self.coord_loss = None

    def update(self, epoch, value, hm, coord):
        if self.best is None:
            self.best = value.avg
            self.hm_loss = hm.avg
            self.coord_loss = coord.avg
            
            logging.info('Best Score Initialize.')
            logging.info(f'Epoch:{epoch}  |  [Best Loss: {self.best:.6f}  |  Hm Loss: {self.hm_loss:.6f}  |  Coord Loss: {self.coord_loss:.6f}]')
            return True
            
        elif self.best > value.avg:
            logging.info(f'Epoch:{epoch}  |  Best Loss: {self.best:.6f}  ----->  {value.avg}')
            self.best = value.avg
            self.hm = hm.avg
            self.coord_loss = coord.avg
            
            logging.info(f'Epoch:{epoch}  |  [Best Loss: {self.best:.6f}  |  Hm Loss: {self.hm_loss:.6f}  |  Coord Loss: {self.coord_loss:.6f}')
            return True

        else:
            return False


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    filename = str(states['epoch']).zfill(3) + "_" + filename
    torch.save(states, os.path.join(output_dir, filename))
    
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))
        
        
def load_checkpoint(states_path):
    try:
        states = torch.load(states_path)
    
        if "epoch" in states:
            epoch = states["epoch"]
        else:
            epoch = None
            
        model = states["state_dict"]
        
        if "optimizer" in states: 
            optimizer = states["optimizer"]
        else:
            optimizer = None
            
        if "scheduler" in states:
            scheduler = states["scheduler"]
        else:
            scheduler = None
        
        logging.info('States Loading...')
        return epoch, model, optimizer, scheduler
    except KeyError as e:
        states = torch.load(states_path)
        
        logging.info('States Loading...')
        return None, states, None, None
    except FileNotFoundError as e:
        logging.info('Model not found. check the directory.')
        logging.info('Starts with initialized model.')
        return None