import numpy as np
from env import sea
from python_vehicle_simulator.lib.control import DPpolePlacement

class DP_PIDagent:
    
    def __init__(self, env):
        
        self.env = env
        self.eta = env.eta
        self.nu = env.nu
        self.sampleTime = env.sample_time
        self.eta3 = np.array([env.eta[0], env.eta[1], env.eta[5]])
        self.nu3 = np.array([env.nu[0], env.nu[1], env.nu[5]])
        
        self.e_int = np.array([0, 0, 0], float)
        self.x_d = 0
        self.y_d = 0
        self.psi_d = 0
        self.wn = np.diag([0.3, 0.3, 0.1]) # PID pole placement
        self.zeta = np.diag([1.0, 1.0, 1.0])
        
        self.x_d_setpoints = []
        self.y_d_setpoints = []
        self.psi_d_setpoints = []
        self.rewards = []
        
    def controlAllocation(self, tau3):
        """
        u_alloc  = controlAllocation(tau3),  tau3 = [tau_X, tau_Y, tau_N]'
        u_alloc = B' * inv( B * B' ) * tau3
        """
        B_pseudoInv = self.env.B.T @ np.linalg.inv(self.env.B @ self.env.B.T)
        u_alloc = np.matmul(B_pseudoInv, tau3)

        return u_alloc
    
    def update(self):
        
        update_dict = {'error': self.e_int,
                       'x_d': self.x_d,
                       'y_d': self.y_d,
                       'psi_d': self.psi_d
        }
        
        return update_dict

    
    def get_action(self, obs):
        
        [self.tau3, self.e_int, self.x_d, self.y_d, self.psi_d] = DPpolePlacement(
            self.e_int, # done
            self.env.M3, # inherited
            self.env.D3, # inherited
            self.eta3, # done
            self.nu3, # done
            self.x_d, # done
            self.y_d, # done 
            self.psi_d, # done
            self.wn, # done
            self.zeta, # done
            self.env.ref, # inherited
            self.sampleTime # done
        )
        
        u_alloc = self.controlAllocation(self.tau3)

        # u_alloc = abs(n) * n --> n = sign(u_alloc) * sqrt(u_alloc)
        # dimU inherited
        n = np.zeros(self.env.dimU)
        for i in range(0, self.env.dimU):
            n[i] = np.sign(u_alloc[i]) * np.sqrt(abs(u_alloc[i]))

        u_control = n

        return u_control
    
    def record(self, reward):
        self.x_d_setpoints.append(self.x_d)
        self.y_d_setpoints.append(self.y_d)
        self.psi_d_setpoints.append(self.psi_d)
        self.rewards.append(reward)
    
    def save(self):
        pass
    def load(self):
        pass
    
class simple_PID:
    def __init__(self):
        pass
    
    def update(self):
        pass
    
    def get_action(self):
        pass
    
    def record(self):
        pass
    
    def save(self):
        pass
    
    def load(self):
        pass
    
    
class SAC:
    def __init__(self):
        pass
    
    def update(self):
        pass
    
    def get_action(self):
        pass
    
    def record(self):
        pass
    
    def save(self):
        pass
    
    def load(self):
        pass