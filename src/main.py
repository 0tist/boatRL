import gym
import os
import numpy as np
from tqdm import tqdm
from env import sea_direct, sea_PID

from manage import make_project_folder, cfg_dict

from stable_baselines3 import DDPG, SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

DOF = 6

# The noise objects for DDPG
# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)

def train(cfg_dict, env):
    cfg_dict = make_project_folder(cfg_dict)
    model = cfg_dict['agent'](cfg_dict['policy'], env, verbose=1)
    model.learn(total_timesteps=cfg_dict['timesteps'], log_interval=10, progress_bar=True)
    model.save(cfg_dict['project_folder']['weights']+f"/{cfg_dict['agent']}")

def run_episode(model, env):
    
    model = cfg_dict['agent'](cfg_dict['policy'], env, verbose=1)
    model.load(cfg_dict['project_folder']['weights']+f"/{cfg_dict['agent']}")
    
    obs = env.reset()
    total_reward = 0
    iters = 0
    rl_reward = []
    sim_data = np.empty( [0, 2*DOF + 2 * env.dimU], float)
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        rl_reward.append(rewards)
        iters += 1
        total_reward += rewards
        sim_data = np.vstack([sim_data, info['sim_data']])
        if dones:   
            simTime = np.arange(start=0, stop=info['sim_time']-env.sample_time, step=env.sample_time)[:, None]
            print(iters)
            print(total_reward)
            break 
        
    np.save(os.path.join(cfg_dict['project_folder']['results'], 'simTime.npy'), simTime)
    np.save(os.path.join(cfg_dict['project_folder']['results'], 'sim_data.npy'), sim_data)
    np.save(os.path.join(cfg_dict['project_folder']['results'], 'rl_rewards.npy'), np.array(rl_reward)) 
    
# from python_vehicle_simulator.lib.plotTimeSeries import *
# print(simTime.shape)
# print(sim_data.shape)
# plotVehicleStates(simTime, sim_data, 1)                    
# plotControls(simTime, sim_data, env, 2)
# numDataPoints = 50                  # number of 3D data points
# FPS = 10                            # frames per second (animated GIF)
# filename = '3D_animation.gif'       # data file for animated GIF
# plot3D(sim_data, numDataPoints, FPS, filename, 3)


if __name__ == '__main__':
    curr_speed, curr_dir, yaw, target = 6, 0, 0, [0, 0, 0]
    # env = sea_direct(curr_speed = curr_speed, curr_dir = curr_dir, yaw = yaw, target = target)
    env = sea_PID(curr_speed = curr_speed, curr_dir = curr_dir, yaw = yaw, target = target)
    train(cfg_dict=cfg_dict, env=env)