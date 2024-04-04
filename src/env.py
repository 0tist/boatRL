# Gym environment of SEA
import numpy as np
from tqdm import tqdm
import gym
from gym import spaces
import math
from python_vehicle_simulator.lib.gnc import sat, attitudeEuler, ssa
from python_vehicle_simulator.lib.control import SimplePID

EPS = 0.00005
INF = 1/EPS

class sea_direct(gym.Env):
    
    def __init__(self, curr_speed, curr_dir, yaw, target:list, max_steps = 2500, with_noise=False):
        
        D2R = math.pi / 180     # deg2rad
        g = 9.81                # acceleration of gravity (m/s^2)
        
        self.max_steps = max_steps
        self.sample_time = 0.02
        self.target = np.array(target)
        self.ref = np.array([target[0], target[1], yaw * D2R], float)
        self.target_yaw = yaw
        self.V_c = curr_speed
        self.with_noise = with_noise
        self.noise_enabled = False
        self.noise_duration = 0
        self.noise_time = 500
        self.beta_c = curr_dir * D2R
        
        self.observations_list = ['x(east)', 'y(north)', 'z(depth)', 'phi(roll)', 'theta(pitch)', 'psi(yaw)',
                                'u(surge)', 'v(sway)', 'w(heave)', 'roll_rate', 'pitch_rate', 'yaw_rate']
        
        self.actions_list = ['bowthruster1', 'bowthruster2', 'rightmainpropeller', 'leftmainpropeller']
        self.controls = self.actions_list
        
        self.observation_space = self._make_obs_space()
        self.action_space = self._make_action_space()
        
        self.simTime = 0
        
        m = 6000.0e3        # mass (kg)
        self.L = 76.2       # Length (m)
        self.T_n = 1.0      # prop. rev. time constant (s)
        
        # Initial Vehicle Params
        self.eta = np.array([0, 0, 0, 0, 0, 0], float) # parameter adjusted after every iter # position/attitude user editable
        self.nu = np.array([0, 0, 0, 0, 0, 0], float) # parameter adjusted after every iter # velocity vector
        self.u_actual = np.array([0, 0, 0, 0], float) # RPM inputs
        
        self.state = np.append(self.eta, self.nu)
        # its the RPM control units 
        self.dimU = len(self.actions_list)
    
        
        #Params for Dynamics of the model
        K = np.diag([2.4, 2.4, 17.6, 17.6])
        T = np.array(
            [[0, 0, 1, 1], [1, 1, 0, 0], [30, 22, -self.L / 2, self.L / 2]], float
        )
        
        self.B = T @ K
        
        Tbis_inv = np.diag([1.0, 1.0, self.L])
        Mbis = np.array(
            [[1.1274, 0, 0], [0, 1.8902, -0.0744], [0, -0.0744, 0.1278]], float
        )
        Dbis = np.array(
            [[0.0358, 0, 0], [0, 0.1183, -0.0124], [0, -0.0041, 0.0308]], float
        )
        
        self.M3 = m * Tbis_inv @ Mbis @ Tbis_inv
        self.M3inv = np.linalg.inv(self.M3)
        self.D3 = m * math.sqrt(g / self.L) * Tbis_inv @ Dbis @ Tbis_inv
    
    def _make_obs_space(self):
        
        '''
        X coordinate - North : can put restrictions to not let the vessel deviate too far
                                no restrictions so far
        Y coordinate - East : Same as X - coordinate
        roll : Vessel shouldn't roll beyond 90 deg, will sink if it does [-pi/2, pi/2]
        pitch : Vessel shouldn't be vertical on the sea? [-pi/2, pi/2]
        yaw : [-pi, pi]
        speed : How fast can the boat go? depends on the thrusters and propeller
                Don't have to worry about it's constraints
        course_angle : can only change the course to a certain angle [-pi, pi],
                        TODO: is this angle relative or global? I think it's global.
        flight_path_angle: 
        '''

        lower_bounds = {
            'x(east)': - INF,
            'y(north)': - INF,
            'z(depth)': - INF,
            'phi(roll)': - np.pi / 2,
            'theta(pitch)': - np.pi / 8,
            'psi(yaw)': - 2 * np.pi,
            'u(surge)': - INF,
            'v(sway)': - INF,
            'w(heave)': - INF,
            'roll_rate': - INF,
            'pitch_rate': - INF,   
            'yaw_rate': - INF
        }
        
        upper_bounds = {
            'x(east)': INF,
            'y(north)': INF,
            'z(depth)': INF,
            'phi(roll)': np.pi / 2,
            'theta(pitch)': np.pi / 8,
            'psi(yaw)': 2 * np.pi,
            'u(surge)': INF,
            'v(sway)': INF,
            'w(heave)': INF,
            'roll_rate': INF,
            'pitch_rate': INF,   
            'yaw_rate': INF
        }
        
        low = np.array([lower_bounds[k] for k in self.observations_list])
        high = np.array([upper_bounds[k] for k in self.observations_list])
        shape = (len(self.observations_list),)
        
        box = spaces.Box(low, high, shape)
        return box
    
    def _make_action_space(self):
        
        '''
        RPM Saturation limits from supply.py
        bowthruster 1 & 2 : [0, 250]
        right and left main propeller: [0, 160]
        '''
        # make action space between -1 and 1
        lower_bounds = {
            'bowthruster1': -250,
            'bowthruster2': -250,
            'rightmainpropeller': -160,
            'leftmainpropeller': -160,
        }
        
        upper_bounds = {
            'bowthruster1': 250,
            'bowthruster2': 250,
            'rightmainpropeller': 160,
            'leftmainpropeller': 160,
        }
        
        low = np.array([lower_bounds[k] for k in self.actions_list])
        high = np.array([upper_bounds[k] for k in self.actions_list])
        shape = (len(self.actions_list),)
        
        box = spaces.Box(low, high, shape)
        return box
        
    def noise_to_waves(self):
        
        if np.random.random() >= 0.7 and not(self.noise_enabled):
            self.noise_enabled = True
            self.vc_noise = 0.5 + (0.1 * np.random.randn())
            self.betac_noise = np.pi/6 + 0.1 * np.random.randn()
            # print(self.betac_noise)
            self.V_c = self.V_c + self.vc_noise
            self.beta_c = self.beta_c + self.betac_noise
            self.noise_duration = 1
        
        elif self.noise_enabled:
            self.noise_duration += 1
            if self.noise_duration >= self.noise_time:
                self.noise_enabled = False
                self.noise_duration = 0
                self.V_c -= self.vc_noise
                self.beta_c -= self.betac_noise
      
    def reset(self):
        
        self.simTime = 0
        self.eta = self.eta = np.array([0, 0, 0, 0, 0, 0], float)
        self.nu = np.array([0, 0, 0, 0, 0, 0], float) 
        self.u_actual = np.array([0, 0, 0, 0], float)
        
        self.steps_left = self.max_steps
        
        self.state = np.append(self.eta, self.nu)
        
        self.noise_enabled = False
        self.noise_duration = 0
        self.vc_noise = 0
        self.betac_noise = 0
        
        return self.state
    
    def _get_info(self):
        return {'l2_norm': np.linalg.norm(self.target-self.eta[:3]),
                'speed': np.linalg.norm(self.nu[:3]),
                'yaw_difference': np.abs(self.target_yaw - self.eta[5]),
                'sim_time': self.simTime,
                'sim_data': np.append( np.append( np.append(self.eta, self.nu), self.u_control), self.u_actual),
                'current_states': (self.V_c, self.vc_noise, self.beta_c, self.betac_noise)
                }
    
    def _get_obs(self):
        return {
            'x(east)': self.eta[0],
            'y(north)': self.eta[1],
            'z(depth)': self.eta[2],
            'phi(roll)': self.eta[3],
            'theta(pitch)': self.eta[4],
            'psi(yaw)': self.eta[5],
            'u(surge)': self.nu[0],
            'v(sway)': self.nu[1],
            'w(heave)': self.nu[2],
            'roll_rate': self.nu[3],
            'pitch_rate': self.nu[4],   
            'yaw_rate': self.nu[5]
        }
        
    
    def dynamics(self, eta, nu, u_actual, u_control, sampleTime):
        
        n = u_actual
        if self.with_noise:
            self.noise_to_waves()
        
        u_c = self.V_c * math.cos(self.beta_c - eta[5])
        v_c = self.V_c * math.sin(self.beta_c - eta[5])
        
        nu_c = np.array([u_c, v_c, 0, 0, 0, 0], float)
        nu_r = nu - nu_c
        
        n_squared = np.zeros(self.dimU)
        for i in range(0, self.dimU):
            n[i] = sat(
                n[i], self.action_space.low[i], self.action_space.high[i]
            )
            n_squared[i] = abs(n[i]) * n[i]
            
        tau3 = np.matmul(self.B, n_squared)
        
        # 3-DOF dynamics
        nu3_r = np.array([nu_r[0], nu_r[1], nu_r[5]])
        nu3_dot = np.matmul(self.M3inv, tau3 - np.matmul(self.D3, nu3_r))
        
        # 6-DOF ship model
        nu_dot = np.array([nu3_dot[0], nu3_dot[1], 0, 0, 0, nu3_dot[2]])
        n_dot = (u_control - u_actual) / self.T_n
        
        # Forward Euler integration
        nu = nu + sampleTime * nu_dot
        n = n + sampleTime * n_dot
        
        u_actual = np.array(n, float)
        
        return nu, u_actual
    
    def calculate_reward(self, eta3):
        k = 1 # proportionality constant
        e = eta3 - self.ref
        e[2] = ssa(e[2])
        
        cartesian_sum = np.linalg.norm(e[:2])
        yaw_error = k / cartesian_sum * e[-1]
        
        reward = 1 / (cartesian_sum + yaw_error + EPS)
        return reward
        
        
    def step(self, action):
        # bt1, bt2, right_mp, left_mp = action
        self.u_control = action
        done = False
        
        self.simTime += self.sample_time
        if self.simTime >= self.sample_time * self.max_steps:
            done = True
        
        self.nu, self.u_actual  = self.dynamics(self.eta, self.nu, self.u_actual, self.u_control, self.sample_time)
        self.eta = attitudeEuler(self.eta, self.nu, self.sample_time)
        
        self.state = np.append(self.eta, self.nu)
        
        # accounts for depth
        if self.eta[2] > 0.1 * self.L:
            done = True
            
        eta3 = np.array([self.eta[0], self.eta[1], self.eta[5]])        
        # cartesian_reward = 1 / (np.linalg.norm(self.eta[:2] - self.target[:2]) + EPS) # or i can take inverse of the distance
        # yaw_reward = 1 / (np.abs(self.eta[5] - self.target_yaw) + EPS)
        
        reward = self.calculate_reward(eta3)
        # with time
        # reward = self.simTime * reward
    
        return self.state, reward, done, self._get_info()
    
    
###################################################
################################################### 
####Action space is the gain of the controllers####
###################################################
################################################### 
 
class sea_PID(gym.Env):
    
    def __init__(self, curr_speed, curr_dir, yaw, target:list, max_steps = 2500, with_noise=False):
        
        D2R = math.pi / 180     # deg2rad
        g = 9.81                # acceleration of gravity (m/s^2)
        
        self.max_steps = max_steps
        self.sample_time = 0.02
        self.target = np.array(target)
        self.ref = np.array([target[0], target[1], yaw * D2R], float)
        self.target_yaw = yaw
        self.V_c = curr_speed
        self.with_noise = with_noise
        self.noise_enabled = False
        self.noise_duration = 0
        self.noise_time = 500
        self.action_frequency = 100
        self.beta_c = curr_dir * D2R
        
        self.observations_list = ['x(east)', 'y(north)', 'z(depth)', 'phi(roll)', 'theta(pitch)', 'psi(yaw)',
                                'u(surge)', 'v(sway)', 'w(heave)', 'roll_rate', 'pitch_rate', 'yaw_rate']
        
        self.actions_list = ['P', 'I', 'D']
        self.controls = self.actions_list
        
        self.observation_space = self._make_obs_space()
        self.action_space = self._make_action_space()
        
        self.simTime = 0
        
        m = 6000.0e3        # mass (kg)
        self.L = 76.2       # Length (m)
        self.n_max = np.array([250, 250, 160, 160], float) #RPM units
        self.T_n = 1.0      # prop. rev. time constant (s)
        
        # Initial Vehicle Params
        self.e_int = np.array([0, 0, 0], float)
        self.x_d = 0.0  # setpoints
        self.y_d = 0.0
        self.psi_d = 0.0
        self.wn = np.diag([0.3, 0.3, 0.1])  # PID pole placement
        self.zeta = np.diag([1.0, 1.0, 1.0])
        
        
        self.eta = np.array([0, 0, 0, 0, 0, 0], float) # parameter adjusted after every iter # position/attitude user editable
        self.nu = np.array([0, 0, 0, 0, 0, 0], float) # parameter adjusted after every iter # velocity vector
        self.u_actual = np.array([0, 0, 0, 0], float) # RPM inputs
        
        self.state = np.append(self.eta, self.nu)
        # these are the dims for RPM units
        self.dimU = 4
    
        
        #Params for Dynamics of the model
        K = np.diag([2.4, 2.4, 17.6, 17.6])
        T = np.array(
            [[0, 0, 1, 1], [1, 1, 0, 0], [30, 22, -self.L / 2, self.L / 2]], float
        )
        
        self.B = T @ K
        
        Tbis_inv = np.diag([1.0, 1.0, self.L])
        Mbis = np.array(
            [[1.1274, 0, 0], [0, 1.8902, -0.0744], [0, -0.0744, 0.1278]], float
        )
        Dbis = np.array(
            [[0.0358, 0, 0], [0, 0.1183, -0.0124], [0, -0.0041, 0.0308]], float
        )
        
        self.M3 = m * Tbis_inv @ Mbis @ Tbis_inv
        self.M3inv = np.linalg.inv(self.M3)
        self.D3 = m * math.sqrt(g / self.L) * Tbis_inv @ Dbis @ Tbis_inv
    
    def _make_obs_space(self):
        
        '''
        X coordinate - North : can put restrictions to not let the vessel deviate too far
                                no restrictions so far
        Y coordinate - East : Same as X - coordinate
        roll : Vessel shouldn't roll beyond 90 deg, will sink if it does [-pi/2, pi/2]
        pitch : Vessel shouldn't be vertical on the sea? [-pi/2, pi/2]
        yaw : [-pi, pi]
        speed : How fast can the boat go? depends on the thrusters and propeller
                Don't have to worry about it's constraints
        course_angle : can only change the course to a certain angle [-pi, pi],
                        TODO: is this angle relative or global? I think it's global.
        flight_path_angle: 
        '''

        lower_bounds = {
            'x(east)': - INF,
            'y(north)': - INF,
            'z(depth)': - INF,
            'phi(roll)': - np.pi / 2,
            'theta(pitch)': - np.pi / 8,
            'psi(yaw)': - 2 * np.pi,
            'u(surge)': - INF,
            'v(sway)': - INF,
            'w(heave)': - INF,
            'roll_rate': - INF,
            'pitch_rate': - INF,   
            'yaw_rate': - INF
        }
        
        upper_bounds = {
            'x(east)': INF,
            'y(north)': INF,
            'z(depth)': INF,
            'phi(roll)': np.pi / 2,
            'theta(pitch)': np.pi / 8,
            'psi(yaw)': 2 * np.pi,
            'u(surge)': INF,
            'v(sway)': INF,
            'w(heave)': INF,
            'roll_rate': INF,
            'pitch_rate': INF,   
            'yaw_rate': INF
        }
        
        low = np.array([lower_bounds[k] for k in self.observations_list])
        high = np.array([upper_bounds[k] for k in self.observations_list])
        shape = (len(self.observations_list),)
        
        box = spaces.Box(low, high, shape)
        return box
    
    def _make_action_space(self):
        
        '''
        PID control units
        '''
        # make action space between -1 and 1
        # lower_bounds = {
        #     'P': 0,
        #     'I': 0,
        #     'D': 0
        # }
        
        # upper_bounds = {
        #     'P': 10,
        #     'I': 10,
        #     'D': 10,
        # }
        
        # low = np.array([lower_bounds[k] for k in self.actions_list])
        # high = np.array([upper_bounds[k] for k in self.actions_list])
        shape = (len(self.actions_list), 3)
        lower_bound = 0
        higher_bound= 10
        box = spaces.Box(low=lower_bound, high=higher_bound, shape=shape)
        
        return box
        
    def noise_to_waves(self):
        
        if np.random.random() >= 0.82 and not(self.noise_enabled):
            self.noise_enabled = True
            self.vc_noise = 0.35 + (0.1 * np.random.randn())
            self.betac_noise = np.pi/9 + 0.1 * np.random.randn()
            # print(self.betac_noise)
            self.V_c = self.V_c + self.vc_noise
            self.beta_c = self.beta_c + self.betac_noise
            self.noise_duration = 1
        
        elif self.noise_enabled:
            self.noise_duration += 1
            if self.noise_duration >= self.noise_time:
                self.noise_enabled = False
                self.noise_duration = 0
                self.V_c -= self.vc_noise
                self.beta_c -= self.betac_noise
      
    def reset(self):
        
        self.simTime = 0
        self.eta = self.eta = np.array([0, 0, 0, 0, 0, 0], float)
        self.nu = np.array([0, 0, 0, 0, 0, 0], float) 
        self.u_actual = np.array([0, 0, 0, 0], float)
        
        self.steps_left = self.max_steps
        
        self.state = np.append(self.eta, self.nu)
        
        self.noise_enabled = False
        self.noise_duration = 0
        self.vc_noise = 0
        self.betac_noise = 0
        
        return self.state
    
    def _get_info(self):
        return {'l2_norm': np.linalg.norm(self.target-self.eta[:3]),
                'speed': np.linalg.norm(self.nu[:3]),
                'yaw_difference': np.abs(self.target_yaw - self.eta[5]),
                'sim_time': self.simTime,
                'sim_data': np.append( np.append( np.append(self.eta, self.nu), self.u_control), self.u_actual),
                'current_states': (self.V_c, self.vc_noise, self.beta_c, self.betac_noise)
                }
    
    def _get_obs(self):
        return {
            'x(east)': self.eta[0],
            'y(north)': self.eta[1],
            'z(depth)': self.eta[2],
            'phi(roll)': self.eta[3],
            'theta(pitch)': self.eta[4],
            'psi(yaw)': self.eta[5],
            'u(surge)': self.nu[0],
            'v(sway)': self.nu[1],
            'w(heave)': self.nu[2],
            'roll_rate': self.nu[3],
            'pitch_rate': self.nu[4],   
            'yaw_rate': self.nu[5]
        }
        
    
    def dynamics(self, eta, nu, u_actual, u_control, sampleTime):
        
        n = u_actual
        if self.with_noise:
            self.noise_to_waves()
        
        u_c = self.V_c * math.cos(self.beta_c - eta[5])
        v_c = self.V_c * math.sin(self.beta_c - eta[5])
        
        nu_c = np.array([u_c, v_c, 0, 0, 0, 0], float)
        nu_r = nu - nu_c
        
        n_squared = np.zeros(self.dimU)
        for i in range(0, self.dimU):
            n[i] = sat(
                n[i], -self.n_max[i], self.n_max[i]
            )
            n_squared[i] = abs(n[i]) * n[i]
            
        tau3 = np.matmul(self.B, n_squared)
        
        # 3-DOF dynamics
        nu3_r = np.array([nu_r[0], nu_r[1], nu_r[5]])
        nu3_dot = np.matmul(self.M3inv, tau3 - np.matmul(self.D3, nu3_r))
        
        # 6-DOF ship model
        nu_dot = np.array([nu3_dot[0], nu3_dot[1], 0, 0, 0, nu3_dot[2]])
        n_dot = (u_control - u_actual) / self.T_n
        
        # Forward Euler integration
        nu = nu + sampleTime * nu_dot
        n = n + sampleTime * n_dot
        
        u_actual = np.array(n, float)
        
        return nu, u_actual
    
    def calculate_reward(self, eta3):
        k = 1 # proportionality constant
        e = eta3 - self.ref
        e[2] = ssa(e[2])
        
        cartesian_sum = np.linalg.norm(e[:2])
        yaw_error = e[2]
        
        reward = 1 / (cartesian_sum + yaw_error + EPS)
        return reward
        
        
    def step(self, action):
        # bt1, bt2, right_mp, left_mp = action
        self.u_control = self.apply_PID(action)
        done = False
        
        self.simTime += self.sample_time
        if self.simTime >= self.sample_time * self.max_steps:
            done = True
        
        self.nu, self.u_actual  = self.dynamics(self.eta, self.nu, self.u_actual, self.u_control, self.sample_time)
        self.eta = attitudeEuler(self.eta, self.nu, self.sample_time)
        
        self.state = np.append(self.eta, self.nu)
        
        # accounts for depth
        if self.eta[2] > 0.1 * self.L:
            done = True
            
        eta3 = np.array([self.eta[0], self.eta[1], self.eta[5]])        
        # cartesian_reward = 1 / (np.linalg.norm(self.eta[:2] - self.target[:2]) + EPS) # or i can take inverse of the distance
        # yaw_reward = 1 / (np.abs(self.eta[5] - self.target_yaw) + EPS)
        
        reward = self.calculate_reward(eta3)
        # with time
        # reward = self.simTime * reward
    
        return self.state, reward, done, self._get_info()
    
    def controlAllocation(self, tau3):
        """
        u_alloc  = controlAllocation(tau3),  tau3 = [tau_X, tau_Y, tau_N]'
        u_alloc = B' * inv( B * B' ) * tau3
        """
        B_pseudoInv = self.B.T @ np.linalg.inv(self.B @ self.B.T)
        u_alloc = np.matmul(B_pseudoInv, tau3)

        return u_alloc
    
    def apply_PID(self, action):
        kp, ki, kd = action
        kp, ki, kd = np.diag(kp), np.diag(ki), np.diag(kd)
        eta3 = np.array([self.eta[0], self.eta[1], self.eta[5]])
        nu3 = np.array([self.nu[0], self.nu[1], self.nu[5]])
        
        [tau3, self.e_int, self.x_d, self.y_d, self.psi_d] = SimplePID(
                self.e_int,
                eta3,
                nu3,
                self.x_d,
                self.y_d,
                self.psi_d,
                self.wn,
                self.ref,
                self.sample_time,
                kp,
                ki,
                kd
            )
        
        u_alloc = self.controlAllocation(tau3)

        # u_alloc = abs(n) * n --> n = sign(u_alloc) * sqrt(u_alloc)
        n = np.zeros(self.dimU)
        for i in range(0, self.dimU):
            n[i] = np.sign(u_alloc[i]) * math.sqrt(abs(u_alloc[i]))

        u_control = n

        return u_control


if __name__ == '__main__':
    curr_speed, curr_dir, yaw, target = 7, 0, 0, [3, 3, 0]
    env = sea_direct(curr_speed = 7, curr_dir = curr_dir, yaw = yaw, target = target)
    print(env.observation_space.shape)
    print(env.action_space.shape)
    obs, reward, done, _ = env.step([50, 50, 50, 50])
    print(obs, reward)
    total_reward = 0
    state = env.reset()
    for i in tqdm(range(100000)):
        action = env.action_space.sample()
        statse, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            # print(total_reward)
            break
        
    print(total_reward)
        # if done:
        #     print(f'Total reward {total_reward}')
        #     break
