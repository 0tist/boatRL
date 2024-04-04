import os
from pathlib import Path
from stable_baselines3 import DDPG, SAC

cfg_dict = {
    'project_folder': {},
    'exp_name': 'mark_PID_SAC_without_noise',
    'policy': 'MlpPolicy',
    'agent': SAC, # or SAC
    'timesteps': 10e6,
}

def make_project_folder(cfg_dict):
    
    exp_folder = os.path.join('./projects/', cfg_dict['exp_name'])
    cfg_dict['project_folder']['parent'] = exp_folder
    os.makedirs(exp_folder)
    os.mkdir(os.path.join(exp_folder, 'weights'))
    cfg_dict['project_folder']['weights'] = os.path.join(exp_folder, 'weights')
    os.mkdir(os.path.join(exp_folder, 'results'))
    cfg_dict['project_folder']['results'] = os.path.join(exp_folder, 'results')
    
    return cfg_dict