import numpy as np
from python_vehicle_simulator import lib

import os
import sys
from pathlib import Path
PARENT_DIR = str(Path(os.getcwd()).parent)
sys.path.append(PARENT_DIR)

from PythonVehicleSimulator.src.python_vehicle_simulator import vehicles
from Genetic_Algorithm_PID_Controller_Tuner.genetic_tuner import lib as ga_lib
from Genetic_Algorithm_PID_Controller_Tuner.genetic_tuner.lib.algorithm import GeneticAlgorithm
from Genetic_Algorithm_PID_Controller_Tuner.genetic_tuner.lib import listtools

config = {
    'population_size' : 100,
    'mutation_probability' : .1,
    'crossover_rate' : .9,
    # maximum simulation runs before finishing
    'max_runs' : 100,
    # maximum timesteps per simulation
    'max_timesteps' : 150,
    # smoothness value of the line in [0, 1]
    'line_smoothness' : .4,
    # Bound for our gain parameters (p, i, d)
    'max_gain_value' : 3,
    # when set to 1, we create a new map this run. When set to 0, loads a new map
    'new_map' : True,
    'runs_per_screenshot' : 10,
    'data_directory' : '/u/97/vasudej1/unix/boatRL/output/ga_pid_data',
    'map_filename' : 'map.csv'
}

population_size = 100

algorithm = GeneticAlgorithm(config)

class chromosome:
    def __init__(self, kp, kd, ki):
        self.kp = kp
        self.kd = kd
        self.ki = ki

# def chromosome(kp, ki, kd):
#     return {'kp': kp,
#             'ki': ki,
#             'kd': kd}

def create_random_population():
    max_gain = 10
    gain_dims = 3
    population = []
    for i in range(population_size):
        chromo = chromosome(
        np.random.random((1, gain_dims)) * max_gain,
        np.random.random((1, gain_dims)) * max_gain,
        np.random.random((1, gain_dims)) * max_gain
        )
        population.append(chromo)
    return population

def create_superior_population(population):
    
    new_population = []
    fitness_values = []
    
    for chromo_idx in range(population_size):
        chromo = population[chromo_idx]
        fitness_values.append(run_sim_for_chromo(chromo))
    
    for chromo_idx in range(population_size):
        parent_idc = algorithm.selection(fitness_values)
        cross_chromo = algorithm.crossover(population[parent_idc[0]], population[parent_idc[1]])
        
        chromo = algorithm.mutation(cross_chromo)
        new_population.append(chromo)
        
    return new_population, fitness_values

def run_sim_for_chromo(chromo):
    N = 10000
    sampleTime = 0.02
    x, y = 50, 50
    yaw = 0
    curr_speed, curr_dir = 0, 0
    control_sys = 'simpleControl'
    vehicle = vehicles.supply(control_sys,
                                  y, x,
                                  yaw,
                                  curr_speed,
                                  curr_dir,
                                  chromo)
    
    distance_list = []
    current_position = 0
    last_distance = 0
    current_summation = 0
    current_distance = 0
    
    DOF = 6                     # degrees of freedom
    t = 0                       # initial simulation time

    # Initial state vectors
    eta = np.array([0, 0, 0, 0, 0, 0], float)    # position/attitude, user editable
    nu = vehicle.nu                              # velocity, defined by vehicle class
    u_actual = vehicle.u_actual                  # actual inputs, defined by vehicle class
    
    # Initialization of table used to store the simulation data
    simData = np.empty( [0, 2*DOF + 2 * vehicle.dimU], float)
    
    for i in range(N):
        time = i
        t = i * sampleTime
        distance_list.append(time)
        
        u_control = vehicle.DPcontrol(eta,nu,sampleTime)
        
        signals = np.append( np.append( np.append(eta,nu),u_control), u_actual )
        simData = np.vstack( [simData, signals] ) 
        [nu, u_actual] = vehicle.dynamics(eta,nu,u_actual,u_control,sampleTime)
        e_int = np.linalg.norm(vehicle.e_int)
        current_distance = 1/e_int
        current_summation = current_summation + current_distance
        
        new_velocity = (chromo.kp*current_distance + chromo.kd*(current_distance - last_distance) +
                        chromo.ki*current_summation)
        
        current_position = current_position + new_velocity

        distance_list[time] = current_distance

        # for the derivative
        last_distance = current_distance
        
        eta = lib.attitudeEuler(eta,nu,sampleTime)
        
    return algorithm.fitness(distance_list)
    
    
    
if __name__ == '__main__':
    
    from tqdm import tqdm
    
    runs = 100
    
    max_scores = []
    avg_scores = []
    kp_vals = []
    ki_vals = []
    kd_vals = []
    
    # map = map.Map(config)
    population = create_random_population()
    
    for i in tqdm(range(runs)):
        
        population, fitness_values = create_superior_population(population)
        
        champs_idx = listtools.max_index_in_list(fitness_values)
        champ_chromo = population[champs_idx]
        
        kp_vals.append(champ_chromo.kp)
        ki_vals.append(champ_chromo.ki)
        kd_vals.append(champ_chromo.kd)
        
        max_scores.append(listtools.max_value_in_list(fitness_values))
        avg_scores.append(listtools.avgList(fitness_values))
        
        print(f"Run {i} : max value {max_scores[i]} : avg value {avg_scores[i]}")
        
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot()
    plt.title("Fitness Values Over Time")

    plt.plot(range(runs), max_scores, label = r"Max Value")
    plt.plot(range(runs), avg_scores, label = r"Average Value")
    plt.legend(loc='lower right')
    plt.xlabel("Run")
    plt.ylabel("Value")
    plt.savefig("results/fitness_values_over_time.png", format="png")

    # plot values of parameters for each run
    plt.figure()
    plt.plot()
    plt.title("Champion Gain Values Per Run")

    plt.plot(range(runs), kp_vals, label = r"Kp")
    plt.plot(range(runs), kd_vals, label = r"Kd")
    plt.plot(range(runs), ki_vals, label = r"Ki")
    plt.legend(loc='center right')
    plt.xlabel("Run")
    plt.ylabel("Value")
    plt.savefig("results/champion_gain_values_per_run.png", format="png")