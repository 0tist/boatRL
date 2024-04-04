import gym 
from gym import spaces, logger
import numpy as np
from scipy.integrate import odeint
from scipy import interpolate
import pandas as pd

class TwoR_Robot(gym.Env):
    def __init__(self, m1=2, m2=2, l1=1, l2=1, path='circle.csv'):
        super(TwoR_Robot,self).__init__()
        # state: q,dq,d2q,q_des,dq_des
        # action: u1,u2/100
        self.observation_space = spaces.Dict({
            'q': spaces.Box(low=-2*np.pi, high=2*np.pi, shape=(2,), dtype=np.float32),
            'dq': spaces.Box(low=-4*np.pi, high=4*np.pi, shape=(2,), dtype=np.float32),
            'q_des': spaces.Box(low=-2*np.pi, high=2*np.pi, shape=(2,), dtype=np.float32),
            'dq_des': spaces.Box(low=-4*np.pi, high=4*np.pi, shape=(2,), dtype=np.float32),
        })
        self.action_space = spaces.Box(low=-1,high=1, shape=(2,))
        self.state = None
        self.steps_beyond_done = None
        self.time = 0
        # length of links
        self.l1 = l1
        self.l2 = l2
        # mass properties
        self.m1 = m1
        self.m2 = m2
        self.izz1 = 1/12*self.m1*self.l1**3
        self.izz2 = 1/12*self.m2*self.l2**3
        
        data = pd.read_csv(path)
        self.t = data['t'].to_numpy()
        self.q_des = data[['q1_des','q2_des']].to_numpy()
        self.dq_des = data[['dq1_des','dq2_des']].to_numpy()
        self.delta = self.t[1]-self.t[0]



    def reset(self):
        q = self.q_des[0] + 0.1*np.random.randn(2)
        self.state = {
            'q': q,
            'dq': self.q_des[0] + 0.05*np.random.randn(2),
            'q_des': self.q_des[0],
            'dq_des': self.dq_des[0],
        }
        self.time = 0
        self.ind = 0
        return self.state

    def fk(self, q1, q2):
        x = self.l1*np.cos(q1)+self.l2*np.cos(q1+q2)
        y = self.l1*np.sin(q1)+self.l2*np.sin(q1+q2)
        return x,y

    def ik(self, x, y):
        q2 = np.arccos((x**2+y**2-self.l1**2-self.l2**2)/(2*self.l1*self.l2))
        q1 = np.arctan2(y,x)-np.arctan2(self.l2*np.sin(q2), self.l1+self.l2*np.cos(q2))
        return q1,q2
    
    
    def get_reward(self, state):
        reward = (state['q']-state['q_des'])**2+0.05*(state['dq']-state['dq_des'])**2
        return -self.delta*np.log(sum(reward))
    
    def xdot(self,q,t,v):
        g = 9.81
        # states
        q1,q2,dq1,dq2 = q     
        
        M11 = self.m2*self.l2**2+self.izz2+self.izz1+self.m1*self.l1**2+self.m2*self.l1**2+2*self.m2*self.l2*self.l1*np.cos(q2)
        M12 = self.m2*self.l2**2+self.m2*self.l2*self.l1*np.cos(q2)+self.izz2
        M21 = self.m2*self.l2**2+self.m2*self.l2*self.l1*np.cos(q2)+self.izz2
        M22 = self.m2*self.l2**2+self.izz2
        M = np.array([[M11, M12], [M21, M22]])
        M_inv = np.linalg.inv(M)
        
        C1 = -np.sin(q2)*dq2*self.l2*self.l1*(2*dq1+dq2)*self.m2
        C2 = self.l2*self.l1*np.sin(q2)*self.m2*dq1**2
        C = np.array([C1,C2]).reshape(-1,1)

        G1 = g*(self.m1*self.l1*np.cos(q1)+self.m2*self.l2*np.cos(q1+q2)+self.m2*self.l1*np.cos(q1))
        G2 = g*np.cos(q1+q2)*self.l2*self.m2
        G = np.array([G1,G2]).reshape(-1,1)

        torque = (v*100).reshape(-1,1)
        xdot1 = np.array([[dq1],[dq2]])
        xdot2 = np.matmul(M_inv,torque-C-G)
        xdot = np.concatenate([xdot1,xdot2],0)
        return xdot.reshape(-1,)
    
    def step(self, action):
        t = np.linspace(self.time, self.time+self.delta, 6)
        current_state = self.state
        q = current_state['q']
        dq = current_state['dq']
        total_q = np.concatenate([q, dq], axis=0)
        out = odeint(self.xdot, total_q,t,args = (action,))
        next_total_q = out[-1]
        next_q = next_total_q[0:2]
        next_dq = next_total_q[2:4]
        self.ind += 1
        self.time += self.delta

        next_state = {
            'q': next_q,
            'dq': next_dq,
            'q_des': self.q_des[self.ind],
            'dq_des': self.dq_des[self.ind],
        }
        
        self.state = next_state
        reward = self.get_reward(self.state)
        done = self.ind == len(self.t)-1
        if not done:
            pass
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            self.reset()
            
        return next_state, reward, done, {}


if __name__ == '__main__':
    env = TwoR_Robot()
    state0 = env.reset()
    state1 = env.step(np.zeros(2))
    print(env.ik(0,0.5))
