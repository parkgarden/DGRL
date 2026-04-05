import gym
import copy
from env.utils.dyn_instance_BA2 import JSP_Instance
from env.utils.dyn_mach_job_op_BA2 import *
from env.utils.dyn_graph_BA2 import Graph

class JSP_Env(gym.Env):
    def __init__(self, args):
        self.args = args
        self.jsp_instance = JSP_Instance(args)

    def step(self, step_ops):
        current_makespan = self.get_makespan()
        
        self.jsp_instance.assign(step_ops)
        avai_ops = self.jsp_instance.current_avai_ops()
        next_makespan = self.get_makespan()
        return avai_ops, current_makespan - next_makespan, self.done()
    
    def reset(self):
        self.jsp_instance.reset()
        return self.jsp_instance.current_avai_ops()
       
    def done(self):
        return self.jsp_instance.done()

    def get_makespan(self):
        return max(m.avai_time() for m in self.jsp_instance.machines)    
    
    def get_graph_data(self):
        return self.jsp_instance.get_graph_data()
        
    def load_instance(self, filename):
        self.jsp_instance.load_instance(filename)
        return self.jsp_instance.current_avai_ops()
    