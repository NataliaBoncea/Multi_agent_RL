import math
import numpy as np

class AutoConfig:
    def __init__(self):
        # --- GLOBAL CONSTANTS ---
        self.WIDTH_OPTIONS = [100, 200, 500] # both width and height
        self.AGENT_COUNT_OPTIONS = [ 3, 5]
        self.TARGET_COUNT_OPTIONS = [1, 3, 5, 10, 15, 20]
        self.START_MODES = ["centralized", "distributed"]
        self.num_episodes = 1000 
        
        # --- BASE MOTION & SENSOR SETUP ---
        self.Vr = 1.0 
        self.Rv = 14.0   
        self.Rs = 1.0    
        self.pd = 0.9    
        self.Da = 3.0    # Avoidance distance 
        self.fa = 0.8    # Collision avoidance factor 
        self.max_yaw = math.pi / 4 # Max yaw angle 
        self.delta_yaw_options = [0, math.radians(20), math.radians(-20)] # Heading adjustments

        # --- REWARD STRUCTURE (Original + Coverage) ---
        self.r1 = 5.0    
        self.r2 = 100.0  
        self.rp = -3.0   
        self.rm = -1.0   
        self.rc = -5.0   
        self.rho = 5.0   
        self.r_cover_scale = 0.01 
        self.r_cover_penalty = -0.1
        self.use_coverage_reward = False # Enabled for the "Improved" architecture

        # Agents
        self.coll_alpha = 1.0
        self.coll_beta = 0.1
        
        # --- HYPERPARAMETERS ---
        self.hidden_size = 64
        self.lr = 0.0005
        self.gamma = 0.95
        self.omega2 = 0.95 
        self.save_freq = 100


        # Epsilon Annealing
        self.epsilon_start = 0.5
        self.epsilon_end = 0.0
        self.epsilon_decay = (self.epsilon_start - self.epsilon_end) / self.num_episodes

    def get_scenario(self, width, agents, mode, targets): # Added targets parameter
        """Generates a complete config instance for a specific experiment run."""
        # Create a dynamic object inheriting all class attributes
        cfg = type('ScenarioConfig', (object,), self.__dict__)()
        
        cfg.WIDTH = width
        cfg.HEIGHT = width
        cfg.Nr = agents
        cfg.Nt = targets # Now correctly assigned from the parameter
        cfg.start_mode = mode
        
        # Updated directory naming to include target count to avoid overwriting
        cfg.ckpt_dir = f"checkpoints/map_{width}_ag_{agents}_tg_{targets}_{mode}"
        cfg.log_file = f"logs/run_{width}_{agents}_tg_{targets}_{mode}.csv"
        
        # Scale max steps by environment size
        if width == 100:
            cfg.max_steps = 400
        elif width == 200:
            cfg.max_steps = 800
        else: # 500x500
            cfg.max_steps = 1500
            
        cfg.num_test_episodes = 100
        return cfg

